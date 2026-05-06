def set_model_internal_data(model, original_data, modified_data, deleted_data):
    """
    Set internal data to model.
    """
    model.__original_data__ = original_data
    list(map(model._prepare_child, model.__original_data__))

    model.__modified_data__ = modified_data
    list(map(model._prepare_child, model.__modified_data__))

    model.__deleted_fields__ = deleted_data

    return model