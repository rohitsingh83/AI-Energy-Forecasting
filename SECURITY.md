# Security and deployment boundaries

This repository is an educational, single-user demonstration using public and
synthetic data. It is **not** a production grid-control or billing application.

- Only load the local joblib bundles you generated or otherwise trust. Pickle /
  joblib deserialization can execute code. The API has no model-upload endpoint.
- The Flask development server has no authentication, authorization, TLS, rate
  limiting, user isolation or production worker management. Keep it local or use
  only non-sensitive data in a controlled demonstration preview.
- Debug mode is off. Run IDs are validated, downloadable files are allowlisted,
  JSON inputs are checked, and request bodies are capped at 16 KiB.
- Browser requests use relative URLs, so the UI works behind a reverse proxy.
  Do not hard-code localhost into browser code for a separate backend.
- The downloader fetches one fixed official HTTPS URL and reads its named ZIP
  member without extracting arbitrary archive paths.
- No SCADA, smart meter, inverter, HVAC actuator, payment service or corporate
  system is connected. Simulation does not dispatch equipment.
- Never commit `.env`, tokens, credentials, private meter data or SSH keys.
- Before any real deployment add a production WSGI server, authentication,
  network controls, logging/monitoring, input schemas, dependencies scanning,
  timezone handling, model/data validation, and a human approval path.

To report a problem in your eventual GitHub copy, use the repository owner's
preferred private reporting channel. No corporate security contact is implied.
