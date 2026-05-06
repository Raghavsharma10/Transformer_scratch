def fix_reserved_word(word, is_module=False):
    """
    Replaces words that may be problematic

    In particular the term 'type' is used in the osid spec, primarily as an argument
    parameter where a type is provided to a method.  'type' is a reserved word
    in python, so we give ours a trailing underscore. If we come across any other
    osid things that are reserved word they can be dealt with here.

    Copied from the builder binder_helpers.py file

    """
    if is_module:
        if word == 'logging':
            word = 'logging_'  # Still deciding this
    else:
        if keyword.iskeyword(word):
            word += '_'
        elif word in ['id', 'type', 'str', 'max', 'input', 'license', 'copyright', 'credits', 'help']:
            word += '_'
    return word