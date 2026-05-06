def convert_to_ssml(text, text_format):
    """
    Convert text to SSML based on the text's current format. NOTE: This module
    is extremely limited at the moment and will be expanded.

    :param text:
        The text to convert.
    :param text_format:
        The text format of the text. Currently supports 'plain', 'html' or None
        for skipping SSML conversion.
    """
    if text_format is None:
        return text
    elif text_format == 'plain':
        return plain_to_ssml(text)
    elif text_format == 'html':
        return html_to_ssml(text)
    else:
        raise ValueError(text_format + ': text format not found.')