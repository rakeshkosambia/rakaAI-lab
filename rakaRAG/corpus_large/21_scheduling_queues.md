# Scheduling and queues

Not every agent run is started by a person. A scheduler starts runs at set times, and a queue holds tasks until a worker is free to take one. The queue smooths a sudden burst of work, so the service is not overwhelmed and the rate limits of the model provider are respected. A concurrency limit caps how many workers run at once. A task that keeps failing is moved to a dead-letter queue, where a person can look at it, instead of blocking everything behind it. Each queued task should carry an idempotency key so that a duplicate delivery does no harm.
