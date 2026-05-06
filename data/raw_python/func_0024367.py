def respond(text=None, ssml=None, attributes=None, reprompt_text=None,
            reprompt_ssml=None, end_session=True):
    """ Build a dict containing a valid response to an Alexa request.

    If speech output is desired, either of `text` or `ssml` should
    be specified.

    :param text: Plain text speech output to be said by Alexa device.
    :param ssml: Speech output in SSML form.
    :param attributes: Dictionary of attributes to store in the session.
    :param end_session: Should the session be terminated after this response?
    :param reprompt_text, reprompt_ssml: Works the same as
        `text`/`ssml`, but instead sets the reprompting speech output.
    """

    obj = {
        'version': '1.0',
        'response': {
            'outputSpeech': {'type': 'PlainText', 'text': ''},
            'shouldEndSession': end_session
        },
        'sessionAttributes': attributes or {}
    }

    if text:
        obj['response']['outputSpeech'] = {'type': 'PlainText', 'text': text}
    elif ssml:
        obj['response']['outputSpeech'] = {'type': 'SSML', 'ssml': ssml}

    reprompt_output = None
    if reprompt_text:
        reprompt_output = {'type': 'PlainText', 'text': reprompt_text}
    elif reprompt_ssml:
        reprompt_output = {'type': 'SSML', 'ssml': reprompt_ssml}

    if reprompt_output:
        obj['response']['reprompt'] = {'outputSpeech': reprompt_output}

    return obj