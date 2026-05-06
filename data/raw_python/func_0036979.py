def load_template(name, directory, extension, encoding, encoding_errors):
    """ Load a template and return its contents as a unicode string. """
    abs_path = get_abs_template_path(name, directory, extension)
    return load_file(abs_path, encoding, encoding_errors)