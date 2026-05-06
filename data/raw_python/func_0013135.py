def source_bash(args, stdin=None):
    """Simply bash-specific wrapper around source-foreign

    Returns a dict to be used as a new environment"""
    args = list(args)
    new_args = ['bash', '--sourcer=source']
    new_args.extend(args)
    return source_foreign(new_args, stdin=stdin)