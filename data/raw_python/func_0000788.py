def autodoc_skip(app, what, name, obj, skip, options):
    """Hook that tells autodoc to include or exclude certain fields.

    Sadly, it doesn't give a reference to the parent object,
    so only the ``name`` can be used for referencing.

    :type app: sphinx.application.Sphinx
    :param what: The parent type, ``class`` or ``module``
    :type what: str
    :param name: The name of the child method/attribute.
    :type name: str
    :param obj: The child value (e.g. a method, dict, or module reference)
    :param options: The current autodoc settings.
    :type options: dict

    .. seealso:: http://www.sphinx-doc.org/en/stable/ext/autodoc.html#event-autodoc-skip-member
    """
    if name in config.EXCLUDE_MEMBERS:
        return True

    if name in config.INCLUDE_MEMBERS:
        return False

    return skip