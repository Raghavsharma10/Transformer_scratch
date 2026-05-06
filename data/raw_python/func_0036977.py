def get_abs_template_path(template_name, directory, extension):
    """ Given a template name, a directory, and an extension, return the
    absolute path to the template. """
    # Get the relative path
    relative_path = join(directory, template_name)
    file_with_ext = template_name

    if extension:
        # If there is a default extension, but no file extension, then add it
        file_name, file_ext = splitext(file_with_ext)
        if not file_ext:
            file_with_ext = extsep.join(
                (file_name, extension.replace(extsep, '')))
            # Rebuild the relative path
            relative_path = join(directory, file_with_ext)

    return abspath(relative_path)