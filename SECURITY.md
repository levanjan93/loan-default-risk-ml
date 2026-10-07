# Security

This project is a **local development / demo** stack. Defaults are chosen for reproducibility, not for internet-facing deployment.

| Component | Default | Guidance |
|-----------|---------|----------|
| Airflow UI (`phase2-ml-pipeline`) | `admin` / `admin` | Change credentials and disable default user before any shared or networked deployment |
| JupyterLab (`phase1-feature-store`) | Token from `JUPYTER_TOKEN` (see `docker-compose.yaml`) | Never run with an empty token on a reachable host |
| Docker Compose | Binds ports `8080`, `8888` to localhost | Do not expose publicly without TLS and auth |

Report issues privately via GitHub Security Advisories on this repository if you discover a vulnerability in project code (not in third-party dependencies).
