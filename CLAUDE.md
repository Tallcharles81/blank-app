# Project conventions

- Never hardcode API keys, database passwords, or secrets in code or committed
  files (including `.env.example`) — use environment variables, with only
  placeholder values in anything checked into git.
- Wrap anything that fetches external data (HTTP requests, API calls) in
  try/except so one failed request doesn't crash the app.
- Add a short comment explaining *why*, not what, for anything non-obvious
  (a workaround, a hidden constraint, a format assumption).
- If unsure about an exact file format or API response shape, check it first
  rather than guessing.
- Commit in small chunks with clear messages so changes are easy to track and
  roll back.
