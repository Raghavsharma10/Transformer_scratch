def print_ctx(ctx):
    """
    Print given context's info.

    :param ctx: Context object.

    :return: None.
    """
    # Print title
    print_title('ctx attributes')

    # Print context object's attributes
    print_text(dir(ctx))

    # Print end title
    print_title('ctx attributes', is_end=True)

    # Print title
    print_title('ctx.options')

    # Print context options dict
    print_text(pformat(vars(ctx.options), indent=4, width=1))

    # Print end title
    print_title('ctx.options', is_end=True)

    # If the context object has `env` attribute.
    # Notice plain context object not has `env` attribute.
    if hasattr(ctx, 'env'):
        # Print title
        print_title('ctx.env')

        # Print context environment variables dict
        print_text(pformat(dict(ctx.env), indent=4, width=1))

        # Print end title
        print_title('ctx.env', is_end=True)