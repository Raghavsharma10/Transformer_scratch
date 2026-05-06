def extract_options_dict(template, options):
    """Extract options from a dictionary against the template"""
    for option, val in template.items():
        if options and option in options:
            yield option, options[option]
        else:
            yield option, Default(template[option]['default'](os.environ))