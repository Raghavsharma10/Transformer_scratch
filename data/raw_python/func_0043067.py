def make_extractor(non_default):
    """
        Return us a function to extract options
        Anything not in non_default is wrapped in a "Default" object
    """
    def extract_options(template, options):
        for option, val in normalise_options(template):
            name = option.replace('-', '_')

            value = getattr(options, name)
            if option not in non_default:
                value = Default(value)

            yield name, value
    return extract_options