def _add_form_fields(obj, lines):
    """Improve the documentation of a Django Form class.

    This highlights the available fields in the form.
    """
    lines.append("**Form fields:**")
    lines.append("")
    for name, field in obj.base_fields.items():
        field_type = "{}.{}".format(field.__class__.__module__, field.__class__.__name__)
        tpl = "* ``{name}``: {label} (:class:`~{field_type}`)"
        lines.append(tpl.format(
            name=name,
            field=field,
            label=field.label or name.replace('_', ' ').title(),
            field_type=field_type
        ))