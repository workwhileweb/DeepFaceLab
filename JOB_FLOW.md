# Job Flow State Machine

## States

- `queued`: Job accepted and persisted in `/data`.
- `running`: Worker process is active and pipeline is executing.
- `cancelling`: Cancel was requested and SIGTERM was sent.
- `completed`: Training and export finished successfully.
- `failed`: Worker exited with error.
- `cancelled`: Job stopped by API cancel request.

## Transitions

- `POST /jobs` -> `queued`
- Queue scanner picks first pending job -> `running`
- Worker success -> `completed`
- Worker error -> `failed`
- `POST /jobs/{id}/cancel` from `queued` -> `cancelled`
- `POST /jobs/{id}/cancel` from `running` -> `cancelling` -> `cancelled`/`failed`
- Service restart: `running` jobs are requeued to `queued`

## Resume behavior

On startup, the service scans existing job folders. Any `running` job without an active worker is moved back to `queued`, then queue scanner restarts it automatically.

## Queue policy (v1)

- Single active worker per service instance (safe default for one GPU).
- FIFO by job directory mtime.
- Service auto-exits when all jobs are terminal and queue stays empty for multiple checks.
