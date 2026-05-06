def what():
    """Print current topics"""

    if not self.isactive():
        lib.echo("No topic")
        sys.exit(lib.USER_ERROR)

    lib.echo(os.environ.get("BE_TOPICS", "This is a bug"))