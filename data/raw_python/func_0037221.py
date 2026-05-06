def check_requirements(to_populate, prompts, helper=False):
    '''
    Iterates through required values, checking to_populate for required values

    If a key in prompts is missing in to_populate and ``helper==True``,
    prompts the user using the values in to_populate. Otherwise, raises an
    error.

    Parameters
    ----------

    to_populate : dict
        Data dictionary to fill. Prompts given to the user are taken from this
        dictionary.

    prompts : dict
        Keys and prompts to use when filling ``to_populate``
    '''

    for kw, prompt in prompts.items():
        if helper:
            if kw not in to_populate:
                to_populate[kw] = click.prompt(prompt)
        else:
            msg = (
                'Required value "{}" not found. '
                'Use helper=True or the --helper '
                'flag for assistance.'.format(kw))

            assert kw in to_populate, msg