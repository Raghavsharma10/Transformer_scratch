def get_model_field_label_and_value(instance, field_name) -> (str, str):
    """
    Returns model field label and value.
    :param instance: Model instance
    :param field_name: Model attribute name
    :return: (label, value) tuple
    """
    label = field_name
    value = str(getattr(instance, field_name))
    for f in instance._meta.fields:
        if f.attname == field_name:
            label = f.verbose_name
            if hasattr(f, 'choices') and len(f.choices) > 0:
                value = choices_label(f.choices, value)
            break
    return label, force_text(value)