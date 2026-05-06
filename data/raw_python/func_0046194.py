def run_and_exit_if(opts, action, *names):
    """
    Run the no-arg function `action` if any of `names` appears in the
    option dict `opts`.
    """
    for name in names:
        if name in opts:
            action()
            sys.exit(0)