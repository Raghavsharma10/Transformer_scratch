def repl():
    """Runs an IPython repl with some context"""

    try:
        import IPython
    except:
        print("ERROR: IPython is not installed. Please install it to use the repl.", file=sys.stderr)
        raise

    IPython.embed(user_ns=dict(
        settings=oz.settings,
        actions=oz._actions,
        uimodules=oz._uimodules,
        routes=oz._routes,
    ))