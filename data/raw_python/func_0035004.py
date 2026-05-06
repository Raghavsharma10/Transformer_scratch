def safe_shell_out(cmd, verbose=False, **kwargs):
    """run cmd and return True if it went ok, False if something went wrong.

    Suppress all output.

    """
    # TODO rename this suppressed_shell_out ?
    # TODO this should probably return 1 if there's an error (i.e. vice-versa).
    # print("cmd %s" % cmd)
    try:
        with open(os.devnull, "w") as fnull:
            with captured_output():
                check_output(cmd, stderr=fnull, **kwargs)
        return True
    except (CalledProcessError, OSError) as e:
        if verbose:
            cprint("    Error running command %s" % ' '.join(cmd), 'err')
            print(e.output)
        return False
    except Exception as e:
        # TODO no idea
        # Can this be if you try and unistall pip? (don't do that)
        return False