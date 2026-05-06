def uri_creator(uri, regex, defaults):
    """Creates url and replaces regex and gives variables"""

    # strip trailing slash
    uri = uri.strip('/')

    # take out variables in uri
    matches = re.findall('{[a-zA-Z0-9\_]+}', uri)
    default_regex = '[a-zA-Z0-9]+'
    
    variables = []

    # iter through matches and replace it with user given regex \
    # if not present, then replace it with default regex
    for match in matches:
        variable = re.sub("{|}", "", match)

        # replace the variable with regex
        set_regex = default_regex
        if variable in regex:
            set_regex = regex[variable]
        
        # set default
        if variable in defaults:
            set_regex = set_regex + "|"

        variables.append(variable)
        
        uri = uri.replace(match, "(" + set_regex + ")")
        
    # debug, put a ^ starts and $ ends with for exact matching
    uri = '^' + uri + '$'
    return {
        'variables' : variables,
        'uri'       : uri
    }