def _get_default_letters(model_admin=None):
    """
    Returns the set of letters defined in the configuration variable
    DEFAULT_ALPHABET. DEFAULT_ALPHABET can be a callable, string, tuple, or
    list and returns a set.

    If a ModelAdmin class is passed, it will look for a DEFAULT_ALPHABET
    attribute and use it instead.
    """
    from django.conf import settings
    import string
    default_ltrs = string.digits + string.ascii_uppercase
    default_letters = getattr(settings, 'DEFAULT_ALPHABET', default_ltrs)
    if model_admin and hasattr(model_admin, 'DEFAULT_ALPHABET'):
        default_letters = model_admin.DEFAULT_ALPHABET
    if callable(default_letters):
        return set(default_letters())
    elif isinstance(default_letters, str):
        return set([x for x in default_letters])
    elif isinstance(default_letters, str):
        return set([x for x in default_letters.decode('utf8')])
    elif isinstance(default_letters, (tuple, list)):
        return set(default_letters)