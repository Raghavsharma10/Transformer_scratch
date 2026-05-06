def main(arguments, toxinidir=None):
    "ctox: tox with conda."
    try:  # pragma: no cover
        # Exit on broken pipe.
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except AttributeError:  # pragma: no cover
        # SIGPIPE is not available on Windows.
        pass

    try:
        import sys
        sys.exit(ctox(arguments, toxinidir))

    except CalledProcessError as c:
        print(c.output)
        return 1

    except NotImplementedError as e:
        gh = "https://github.com/hayd/ctox/issues"
        from colorama import Style
        cprint(Style.BRIGHT + str(e), 'err')
        cprint("If this is a valid tox.ini substitution, please open an issue on\n"
               "github and request support: %s." % gh, 'warn')
        return 1

    except KeyboardInterrupt:  # pragma: no cover
        return 1