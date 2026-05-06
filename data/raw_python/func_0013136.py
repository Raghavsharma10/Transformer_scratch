def source_zsh(args, stdin=None):
    """Simply zsh-specific wrapper around source-foreign

    Returns a dict to be used as a new environment"""
    args = list(args)
    new_args = ['zsh', '--sourcer=source']
    new_args.extend(args)
    return source_foreign(new_args, stdin=stdin)