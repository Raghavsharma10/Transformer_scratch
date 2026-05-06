def recover_hashmap_model_from_data(model_class, original_data, modified_data, deleted_data, field_type):
    """
    Function to reconstruct a model from DirtyModel basic information: original data, the modified and deleted
    fields.
    Necessary for pickle an object
    """
    model = model_class(field_type=field_type[0](**field_type[1]))
    return set_model_internal_data(model, original_data, modified_data, deleted_data)