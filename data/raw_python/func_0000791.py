def _add_model_fields_as_params(app, obj, lines):
    """Improve the documentation of a Django model subclass.

    This adds all model fields as parameters to the ``__init__()`` method.

    :type app: sphinx.application.Sphinx
    :type lines: list
    """
    for field in obj._meta.get_fields():
        try:
            help_text = strip_tags(force_text(field.help_text))
            verbose_name = force_text(field.verbose_name).capitalize()
        except AttributeError:
            # e.g. ManyToOneRel
            continue

        # Add parameter
        if help_text:
            lines.append(u':param %s: %s' % (field.name, help_text))
        else:
            lines.append(u':param %s: %s' % (field.name, verbose_name))

        # Add type
        lines.append(_get_field_type(field))

    if 'sphinx.ext.inheritance_diagram' in app.extensions and \
            'sphinx.ext.graphviz' in app.extensions and \
            not any('inheritance-diagram::' in line for line in lines):
        lines.append('.. inheritance-diagram::')