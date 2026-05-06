def deprecated(replacement_description):
    """States that method is deprecated.

    :param replacement_description: Describes what must be used instead.
    :return: the original method with modified docstring.

    """

    def decorate(fn_or_class):
        if isinstance(fn_or_class, type):
            pass  # Can't change __doc__ of type objects
        else:
            try:
                fn_or_class.__doc__ = "This API point is obsolete. %s\n\n%s" % (
                    replacement_description,
                    fn_or_class.__doc__,
                )
            except AttributeError:
                pass  # For Cython method descriptors, etc.
        return fn_or_class

    return decorate