def click_info_switch(f):
    """Decorator to create eager Click info switch option, as described in:
    http://click.pocoo.org/6/options/#callbacks-and-eager-options.

    Takes a no-argument function and abstracts the boilerplate required by
    Click (value checking, exit on done).

    Example:

        @click.option('--my-option', is_flag=True, callback=my_option,
                    expose_value=False, is_eager=True)
        def test():
            pass

        @click_info_switch
        def my_option()
            click.echo('some info related to my switch')
    """

    @wraps(f)
    def wrapped(ctx, param, value):
        if not value or ctx.resilient_parsing:
            return
        f()
        ctx.exit()
    return wrapped