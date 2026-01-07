from __future__ import annotations

import argparse

import uvicorn

from translation_app.web.server import _configure_logging, create_app


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Translation Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        create_app(),
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level="info",
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
