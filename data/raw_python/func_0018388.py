def main():
    """Be the top-level entrypoint. Return a shell status code."""
    commands = {'hash': peep_hash,
                'install': peep_install,
                'port': peep_port}
    try:
        if len(argv) >= 2 and argv[1] in commands:
            return commands[argv[1]](argv[2:])
        else:
            # Fall through to top-level pip main() for everything else:
            return pip.main()
    except PipException as exc:
        return exc.error_code