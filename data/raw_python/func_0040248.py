def resolve(args):
    """Just print the result of parsing a target string."""
    if not args:
        log.error('Exactly 1 argument is required.')
        app.quit(1)
    print(address.new(args[0]))