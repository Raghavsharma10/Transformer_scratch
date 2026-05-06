def convert_unicode_2_utf8(input):
    '''Return a copy of `input` with every str component encoded from unicode to
    utf-8.
    '''
    if isinstance(input, dict):
        try:
            # python-2.6
            return dict((convert_unicode_2_utf8(key), convert_unicode_2_utf8(value))
                        for key, value
                        in input.iteritems())
        except AttributeError:
            # since python-2.7 cf. http://stackoverflow.com/a/1747827
            # [the ugly eval('...') is required for a valid syntax on
            # python-2.6, cf. http://stackoverflow.com/a/25049535]
            return eval('''{convert_unicode_2_utf8(key): convert_unicode_2_utf8(value)
                           for key, value
                           in input.items()}''')
    elif isinstance(input, list):
        return [convert_unicode_2_utf8(element) for element in input]
    # elif order relevant: python2 vs. python3
    # cf. http://stackoverflow.com/a/19877309
    elif isinstance(input, str):
        return input
    else:
        try:
            if eval('''isinstance(input, unicode)'''):
                return input.encode('utf-8')
        except NameError:
            # unicode does not exist in python-3.x
            pass
        return input