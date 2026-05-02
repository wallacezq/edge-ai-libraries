# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference scheduler for continuous batching of VLM requests.

This module implements a request queue and scheduler loop that uses
``ContinuousBatchingPipeline.add_request()`` to support concurrent
inference requests without hitting the "Generate cannot be called while
ContinuousBatchingPipeline is already in running state" error.

When the pipeline does not expose ``add_request`` (e.g. older openvino-genai
versions), requests are serialized through the scheduler thread so that only
one ``generate()`` call is active at any time while still allowing concurrent
API clients to queue work.
"""

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

from src.utils.common import logger


@dataclass
class InferenceRequest:
    """A single inference request submitted to the scheduler."""

    request_id: str
    generation_kwargs: Dict[str, Any]
    stream: bool = False
    # Populated by the scheduler after completion
    result: Any = None
    error: Optional[Exception] = None
    # Signaled when the request is complete (for non-streaming)
    done_event: threading.Event = field(default_factory=threading.Event)
    # For streaming: a queue that receives token chunks
    token_queue: Optional[Queue] = field(default=None)
    # Callback for perf_metrics after generation
    perf_metrics: Any = None


_STREAM_SENTINEL = object()


class InferenceScheduler:
    """Manages concurrent VLM inference via continuous batching or serialized generation.

    The scheduler runs a background thread that processes queued requests. It
    attempts to use the pipeline's ``add_request()`` / ``step()`` loop for true
    continuous batching. If the pipeline doesn't support those methods, it falls
    back to sequential ``generate()`` calls.

    Usage::

        scheduler = InferenceScheduler(pipe)
        scheduler.start()

        # Submit a request (from an async context)
        result = await scheduler.submit(generation_kwargs, stream=False)

        # For streaming
        token_queue = await scheduler.submit(generation_kwargs, stream=True)
        for token in iter(token_queue.get, SENTINEL):
            ...
    """

    def __init__(self, pipe, *, max_queue_size: int = 64):
        self._pipe = pipe
        self._request_queue: Queue = Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._supports_add_request = hasattr(pipe, "add_request") and callable(
            getattr(pipe, "add_request", None)
        )
        self._supports_step = hasattr(pipe, "step") and callable(
            getattr(pipe, "step", None)
        )
        self._use_continuous_batching = (
            self._supports_add_request and self._supports_step
        )

        if self._use_continuous_batching:
            logger.info(
                "InferenceScheduler: pipeline supports add_request/step — "
                "enabling continuous batching mode."
            )
        else:
            logger.info(
                "InferenceScheduler: pipeline does not expose add_request/step — "
                "using serialized generation mode."
            )

    @property
    def use_continuous_batching(self) -> bool:
        return self._use_continuous_batching

    def start(self):
        """Start the scheduler background thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._run_loop, name="InferenceScheduler", daemon=True
        )
        self._worker_thread.start()
        logger.info("InferenceScheduler started.")

    def stop(self):
        """Stop the scheduler gracefully."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        logger.info("InferenceScheduler stopped.")

    async def submit(
        self, generation_kwargs: Dict[str, Any], *, stream: bool = False
    ):
        """Submit an inference request and wait for the result.

        Args:
            generation_kwargs: Keyword arguments passed to ``pipe.generate()``.
            stream: If True, returns a Queue that yields token strings followed
                by a sentinel value.

        Returns:
            For non-streaming: the generation result object.
            For streaming: a tuple of (token_queue, request) where token_queue
                yields string chunks and the request object holds perf_metrics
                after completion.
        """
        request_id = str(uuid.uuid4())
        token_queue = Queue() if stream else None

        req = InferenceRequest(
            request_id=request_id,
            generation_kwargs=generation_kwargs,
            stream=stream,
            token_queue=token_queue,
        )

        self._request_queue.put(req)
        logger.debug(
            "Request %s queued (stream=%s). Queue depth: %d",
            request_id,
            stream,
            self._request_queue.qsize(),
        )

        if stream:
            return token_queue, req
        else:
            # Wait for completion in a non-blocking way
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, req.done_event.wait)
            if req.error:
                raise req.error
            return req.result

    def _run_loop(self):
        """Main scheduler loop running in a background thread."""
        if self._use_continuous_batching:
            self._run_continuous_batching_loop()
        else:
            self._run_serialized_loop()

    def _run_continuous_batching_loop(self):
        """Process requests using add_request/step for true continuous batching."""
        active_requests: Dict[str, InferenceRequest] = {}

        while self._running:
            # Drain the queue — add all pending requests to the pipeline
            while not self._request_queue.empty():
                try:
                    req = self._request_queue.get_nowait()
                except Empty:
                    break

                try:
                    kwargs = dict(req.generation_kwargs)
                    generation_config = kwargs.pop("generation_config", None)
                    prompt = kwargs.pop("prompt", "")
                    images = kwargs.pop("images", None)
                    videos = kwargs.pop("videos", None)

                    add_kwargs = {}
                    if images is not None:
                        add_kwargs["images"] = images
                    if videos is not None:
                        add_kwargs["videos"] = videos

                    if generation_config is not None:
                        self._pipe.add_request(
                            req.request_id,
                            prompt,
                            generation_config,
                            **add_kwargs,
                        )
                    else:
                        self._pipe.add_request(
                            req.request_id, prompt, **add_kwargs
                        )

                    active_requests[req.request_id] = req
                    logger.debug(
                        "Added request %s to continuous batching pipeline.",
                        req.request_id,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to add request %s: %s", req.request_id, e
                    )
                    req.error = e
                    req.done_event.set()
                    if req.token_queue:
                        req.token_queue.put(_STREAM_SENTINEL)

            # Run one step of the pipeline
            if active_requests:
                try:
                    outputs = self._pipe.step()
                except Exception as e:
                    logger.error("Pipeline step() failed: %s", e)
                    # Mark all active requests as failed
                    for req in active_requests.values():
                        req.error = e
                        req.done_event.set()
                        if req.token_queue:
                            req.token_queue.put(_STREAM_SENTINEL)
                    active_requests.clear()
                    continue

                # Process step outputs
                if outputs:
                    self._process_step_outputs(outputs, active_requests)

                # Check for finished requests
                if hasattr(self._pipe, "has_non_finished_requests"):
                    if not self._pipe.has_non_finished_requests() and not active_requests:
                        time.sleep(0.001)
            else:
                # No active requests, wait briefly for new ones
                time.sleep(0.005)

    def _process_step_outputs(
        self, outputs, active_requests: Dict[str, InferenceRequest]
    ):
        """Route step() outputs to the appropriate request handlers."""
        # The format of outputs depends on the openvino_genai version.
        # Common patterns:
        # - List of GenerationOutput objects with request_id, token_ids, finished
        # - List of GenerationResult objects
        if not outputs:
            return

        for output in outputs:
            req_id = getattr(output, "request_id", None)
            if req_id is None:
                continue

            req = active_requests.get(req_id)
            if req is None:
                continue

            # Stream tokens if available
            token_text = None
            if hasattr(output, "token_text"):
                token_text = output.token_text
            elif hasattr(output, "text"):
                token_text = output.text
            elif hasattr(output, "generated_text"):
                token_text = output.generated_text

            if token_text and req.stream and req.token_queue:
                req.token_queue.put(token_text)

            # Check if this request is finished
            finished = getattr(output, "finished", False) or getattr(
                output, "is_finished", False
            )
            if finished:
                if req.stream:
                    req.perf_metrics = getattr(output, "perf_metrics", None)
                    req.token_queue.put(_STREAM_SENTINEL)
                else:
                    req.result = output
                    req.perf_metrics = getattr(output, "perf_metrics", None)
                req.done_event.set()
                del active_requests[req_id]
                logger.debug("Request %s completed.", req_id)

    def _run_serialized_loop(self):
        """Fallback: process requests one at a time using generate()."""
        while self._running:
            try:
                req = self._request_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                if req.stream:
                    self._run_streaming_request(req)
                else:
                    self._run_blocking_request(req)
            except Exception as e:
                logger.error(
                    "Error processing request %s: %s", req.request_id, e
                )
                req.error = e
                req.done_event.set()
                if req.token_queue:
                    req.token_queue.put(_STREAM_SENTINEL)

    def _run_blocking_request(self, req: InferenceRequest):
        """Execute a non-streaming generate() call."""
        try:
            result = self._pipe.generate(**req.generation_kwargs)
            req.result = result
            req.perf_metrics = getattr(result, "perf_metrics", None)
        except Exception as e:
            req.error = e
        finally:
            req.done_event.set()

    def _run_streaming_request(self, req: InferenceRequest):
        """Execute a streaming generate() call, pushing tokens to the queue."""
        try:
            kwargs = dict(req.generation_kwargs)

            # Create a streamer callback that pushes to the token queue
            def token_callback(text: str):
                if text:
                    req.token_queue.put(text)

            kwargs["streamer"] = token_callback
            result = self._pipe.generate(**kwargs)
            req.perf_metrics = getattr(result, "perf_metrics", None)
        except Exception as e:
            req.error = e
        finally:
            req.token_queue.put(_STREAM_SENTINEL)
            req.done_event.set()


# Module-level sentinel for consumers to detect end-of-stream
STREAM_DONE = _STREAM_SENTINEL
