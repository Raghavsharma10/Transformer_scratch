def improve_model_docstring(app, what, name, obj, options, lines):
    """Hook that improves the autodoc docstrings for Django models.

    :type app: sphinx.application.Sphinx
    :param what: The parent type, ``class`` or ``module``
    :type what: str
    :param name: The dotted path to the child method/attribute.
    :type name: str
    :param obj: The Python object that i s being documented.
    :param options: The current autodoc settings.
    :type options: dict
    :param lines: The current documentation lines
    :type lines: list
    """
    if what == 'class':
        _improve_class_docs(app, obj, lines)
    elif what == 'attribute':
        _improve_attribute_docs(obj, name, lines)
    elif what == 'method':
        _improve_method_docs(obj, name, lines)

    # Return the extended docstring
    return lines