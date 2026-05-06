def recover_fast_dynamic_model_from_data(model_class, original_data, modified_data, deleted_data, field_types):
    """
    Function to reconstruct a model from DirtyModel basic information: original data, the modified and deleted
    fields.
    Necessary for pickle an object
    """
    model = model_class()

    model.__field_types__ = {k: d[0](**d[1]) for k, d in field_types.items()}

    return set_model_internal_data(model, original_data, modified_data, deleted_data)