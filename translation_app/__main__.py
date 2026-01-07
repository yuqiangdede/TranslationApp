"""Package entrypoint: `python -m translation_app` launches the Web UI."""

from translation_app.web.server import main

if __name__ == "__main__":
    raise SystemExit(main())
