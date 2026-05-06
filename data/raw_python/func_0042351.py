def fetch_model_data(model_querysets, model_ids_to_fetch):
    """
    Given a dictionary of models to querysets and model IDs to models, fetch the IDs
    for every model and return the objects in the following structure.

    {
        model: {
            id: obj,
            ...
        },
        ...
    }
    """
    return {
        model: id_dict(model_querysets[model].filter(id__in=ids_to_fetch))
        for model, ids_to_fetch in model_ids_to_fetch.items()
    }