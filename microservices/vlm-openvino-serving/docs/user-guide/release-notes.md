# Release Notes

## Unreleased

### New Features

- **Continuous Batching Support**: Added an inference scheduler (`InferenceScheduler`) that queues concurrent requests and processes them safely. When the pipeline supports `add_request()`/`step()` (openvino-genai ≥ 2025.4), true continuous batching is used. Otherwise, requests are serialized automatically to prevent the "Generate cannot be called while ContinuousBatchingPipeline is already in running state" error.

### New Environment Variables

- `VLM_CONTINUOUS_BATCHING` (default: `true`): Enable/disable the inference scheduler.
- `VLM_SCHEDULER_MAX_QUEUE` (default: `64`): Maximum pending requests in the scheduler queue.

## Releases 1.2.0, 1.2.1, 1.2.2, 1.2.3, 1.3.0 and 1.3.1

This microservice supports features based on the requirements of Video Search and Summarization sample application which is using this microservice. Refer to Video Search and Summarization [release notes](../../../../sample-applications/video-search-and-summarization/docs/user-guide/release-notes.md) for release details of this microservice.
