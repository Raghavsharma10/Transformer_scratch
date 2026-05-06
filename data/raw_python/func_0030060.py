def construct_re(url_template, match_whole_str=False, converters=None,
                 default_converter='string', anonymous=False):
    '''
    url_template - str or unicode representing template

    Constructed pattern expects urlencoded string!

    returns  (compiled re pattern, 
              dict {url param name: [converter name, converter args (str)]},
              list of (variable name, converter name, converter args name))

    If anonymous=True is set, regexp will be compiled without names of variables.
    This is handy for example, if you want to dump an url map to JSON.
    '''
    # needed for reverse url building (or not needed?)
    builder_params = []
    # found url params and their converters
    url_params = {}
    result = r'^'
    parts = _split_pattern.split(url_template)
    for i, part in enumerate(parts):
        is_url_pattern = _static_url_pattern.match(part)
        if is_url_pattern:
            #NOTE: right order:
            #      - make part str if it was unicode
            #      - urlquote part
            #      - escape all specific for re chars in part
            result += re.escape(urlquote(part))
            builder_params.append(part)
            continue
        is_converter = _converter_pattern.match(part)
        if is_converter:
            groups = is_converter.groupdict()
            converter_name = groups['converter'] or default_converter
            conv_object = init_converter(converters[converter_name],
                                         groups['args'])
            variable = groups['variable']
            builder_params.append((variable, conv_object))
            url_params[variable] = conv_object
            if anonymous:
                result += conv_object.regex
            else:
                result += '(?P<{}>{})'.format(variable, conv_object.regex)
            continue
        raise ValueError('Incorrect url template {!r}'.format(url_template))
    if match_whole_str:
        result += '$'
    return re.compile(result), url_params, builder_params