def preprocess(string):
    """
    Preprocesses a string, by replacing ${VARNAME} with
    os.environ['VARNAME']

    Parameters
    ----------
    string: the str object to preprocess

    Returns
    -------
    the preprocessed string
    """

    split = string.split('${')

    rval = [split[0]]

    for candidate in split[1:]:
        subsplit = candidate.split('}')

        if len(subsplit) < 2:
            raise ValueError('Open ${ not followed by } before ' \
                    + 'end of string or next ${ in "' \
                    + string + '"')

        varname = subsplit[0]

        if varname == 'PYLEARN2_TRAIN_FILE_NAME':
            warnings.warn("PYLEARN2_TRAIN_FILE_NAME is deprecated and may be "
                    "removed from the library on or after Oct 22, 2013. Switch"
                    " to PYLEARN2_TRAIN_FILE_FULL_STEM")

        try:
            val = os.environ[varname]
        except KeyError:
            if varname == 'PYLEARN2_DATA_PATH':
                raise NoDataPathError()
            if varname == 'PYLEARN2_VIEWER_COMMAND':
                raise EnvironmentVariableError(environment_variable_essay)

            raise ValueError('Unrecognized environment variable "' + varname
                    + '". Did you mean ' + match(varname, os.environ.keys())
                    + '?')

        rval.append(val)

        rval.append('}'.join(subsplit[1:]))

    rval = ''.join(rval)

    return rval