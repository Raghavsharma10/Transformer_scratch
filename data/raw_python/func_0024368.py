def reprompt(text=None, ssml=None, attributes=None):
    """Convenience method to save a little bit of typing for the common case of
    reprompting the user. Simply calls :py:func:`alexandra.util.respond` with
    the given arguments and holds the session open.

    One of either the `text` or `ssml` should be provided if any
    speech output is desired.

    :param text: Plain text speech output
    :param ssml: Speech output in SSML format
    :param attributes: Dictionary of attributes to store in the current session
    """

    return respond(
        reprompt_text=text,
        reprompt_ssml=ssml,
        attributes=attributes,
        end_session=False
    )