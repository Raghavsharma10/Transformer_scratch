def absolute_path(user_path):
    """
    Some paths must be made absolute, this will attempt to convert them.
    """
    if os.path.abspath(user_path):
        return unix_path_coercion(user_path)
    else:
        try:
            openaccess_epub.utils.evaluate_relative_path(relative=user_path)
        except:
            raise ValidationError('This path could not be rendered as absolute')