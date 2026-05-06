def watson_request(text, synth_args):
    """
    Makes a single request to the IBM Watson text-to-speech API.

    :param text:
        The text that will be synthesized to audio.
    :param synth_args:
        A dictionary of arguments to add to the request. These should include
        username and password for authentication.
    """
    params = {
        'text': text,
        'accept': 'audio/wav'
    }
    if synth_args is not None:
        params.update(synth_args)

    if 'username' in params:
        username = params.pop('username')
    else:
        raise Warning('The IBM Watson API requires credentials that should be passed as "username" and "password" in "synth_args"')
    if 'password' in params:
        password = params.pop('password')
    else:
        raise Warning('The IBM Watson API requires credentials that should be passed as "username" and "password" in "synth_args"')

    return requests.get(watson_url, auth=(username, password), params=params)