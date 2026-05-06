def encode_properties(parameters):
    """
    Performs encoding of url parameters from dictionary to a string. It does
    not escape backslash because it is not needed.

    See: http://www.jfrog.com/confluence/display/RTF/Artifactory+REST+API#ArtifactoryRESTAPI-SetItemProperties
    """
    result = []

    for param in iter(sorted(parameters)):
        if isinstance(parameters[param], (list, tuple)):
            value = ','.join([escape_chars(x) for x in parameters[param]])
        else:
            value = escape_chars(parameters[param])

        result.append("%s=%s" % (param, value))

    return '|'.join(result)