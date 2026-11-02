from .cli import main

if __name__ == "__main__":  # spawn-based worker processes re-import this module as __mp_main__
    raise SystemExit(main())
