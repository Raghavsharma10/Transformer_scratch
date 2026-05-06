def format(fmt_str=None, plural=False, bestprefix=False):
    """Context manager for printing bitmath instances.

``fmt_str`` - a formatting mini-language compat formatting string. See
the @properties (above) for a list of available items.

``plural`` - True enables printing instances with 's's if they're
plural. False (default) prints them as singular (no trailing 's').

``bestprefix`` - True enables printing instances in their best
human-readable representation. False, the default, prints instances
using their current prefix unit.
    """
    if 'bitmath' not in globals():
        import bitmath

    if plural:
        orig_fmt_plural = bitmath.format_plural
        bitmath.format_plural = True

    if fmt_str:
        orig_fmt_str = bitmath.format_string
        bitmath.format_string = fmt_str

    yield

    if plural:
        bitmath.format_plural = orig_fmt_plural

    if fmt_str:
        bitmath.format_string = orig_fmt_str