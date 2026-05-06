def recover_dynamic_model_from_data(model_class, original_data, modified_data, deleted_data, structure):
    """
    Function to reconstruct a model from DirtyModel basic information: original data, the modified and deleted
    fields.
    Necessary for pickle an object
    """
    model = model_class()

    model.__structure__ = {k: d[0](**d[1]) for k, d in structure.items()}

    return set_model_internal_data(model, original_data, modified_data, deleted_data)