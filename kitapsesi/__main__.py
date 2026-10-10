import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        from .cli import main as cli_main
        raise SystemExit(cli_main(sys.argv[2:]))
    from .gui import run
    raise SystemExit(run())


main()
