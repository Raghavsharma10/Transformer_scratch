def shortentext(text, minlength, placeholder='...'):
    """
    Shorten some text by replacing the last part with a placeholder (such as '...')

    :type text: string
    :param text: The text to shorten

    :type minlength: integer
    :param minlength: The minimum length before a shortening will occur

    :type placeholder: string
    :param placeholder: The text to append after removing protruding text.
    """

    return textwrap.shorten(text, minlength, placeholder=str(placeholder))