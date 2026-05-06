def parse_string(s):
        '''
        Parses a foreign resource URL into the URL string itself and any
        relevant args and kwargs
        '''
        matched_obj = SPLIT_URL_RE.match(s)
        if not matched_obj:
            raise URLParseException('Invalid Resource URL: "%s"' % s)

        url_string, arguments_string = matched_obj.groups()
        args_as_strings = URL_ARGUMENTS_RE.findall(arguments_string)

        # Determine args and kwargs
        args = []
        kwargs = {}
        for arg_string in args_as_strings:
            kwarg_match = ARG_RE.match(arg_string)
            if kwarg_match:
                key, value = kwarg_match.groups()
                kwargs[key.strip()] = value.strip()
            else:
                args.append(arg_string.strip())

        # Default to HTTP if url_string has no URL
        if not SCHEME_RE.match(url_string):
            url_string = '%s://%s' % (DEFAULT_SCHEME, url_string)

        return url_string.strip(), args, kwargs