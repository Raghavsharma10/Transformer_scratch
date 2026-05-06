def _summarize_object_type(model):
    """
        This function returns the summary for a given model
    """
    # the fields for the service's model
    model_fields = {field.name: field for field in list(model.fields())}
    # summarize the model
    return {
        'fields': [{
            'name': key,
            'type': type(convert_peewee_field(value)).__name__
            } for key, value in model_fields.items()
        ]
    }