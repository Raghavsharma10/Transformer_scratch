def common_options(func):
    """Commonly used command options."""

    def parse_preset(ctx, param, value):
        return PRESETS.get(value, (None, None))

    def parse_private(ctx, param, value):
        return hex_from_b64(value) if value else None

    func = click.option('--private', default=None, help='Private.', callback=parse_private)(func)

    func = click.option(
        '--preset',
        default=None, help='Preset ID defining prime and generator pair.',
        type=click.Choice(PRESETS.keys()), callback=parse_preset
    )(func)

    return func