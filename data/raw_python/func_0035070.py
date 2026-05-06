def ModelOption(model):
    """Returns *model* if a valid choice.

    Returns the string if it specifies a ``YNGKP_`` model variant.

    Returns *('ExpCM', prefsfile)* if it specifies an ``ExpCM_`` model.
    """
    yngkpmatch = re.compile('^YNGKP_M[{0}]$'.format(''.join([m[1 : ] for m in yngkp_modelvariants])))
    if yngkpmatch.search(model):
        return model
    elif len(model) > 6 and model[ : 6] == 'ExpCM_':
        fname = model[6 : ]
        if os.path.isfile(fname):
            return ('ExpCM', fname)
        else:
            raise ValueError("ExpCM_ must be followed by the name of an existing file. You specified the following, which is not an existing file: %s" % fname)
    else:
        raise ValueError("Invalid model")