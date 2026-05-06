def serialize(value, field):
    """
    Form values serialization

    :param object value: A value to be serialized\
    for saving it into the database and later\
    loading it into the form as initial value
    """
    assert isinstance(field, forms.Field)
    if isinstance(field, forms.ModelMultipleChoiceField):
        return json.dumps([v.pk for v in value])
    # todo: remove
    if isinstance(value, models.Model):
        return value.pk
    return value