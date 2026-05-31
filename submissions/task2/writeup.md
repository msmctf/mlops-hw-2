# Task 2 Writeup

**1. What's wrong with a local-file audit log in a real production deployment? Name one concrete failure mode.**
A local-file audit log is vulnerable to data loss and lacks concurrency controls. A concrete failure mode is a race condition: if two engineers attempt to promote different models at the exact same moment from different terminals (or load-balanced CI/CD workers), their scripts could try to append to the file simultaneously, causing interleaved/corrupted JSON lines or overwriting each other's entries.

**2. If you were extending this CLI to production use, name one feature you'd add and why.**
I would add a required `--message` or `--reason` flag to the `set` and `rollback` commands (e.g., `set production v6 --message "Deploying v6 with prompt hardening"`). This would attach context and intent to the audit log event, which is invaluable for post-mortems or when another team member asks "why did you roll back the model at 3 AM?".
