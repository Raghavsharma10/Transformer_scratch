def _summarize_o_mutation_type(model):
    """
        This function create the actual mutation io summary corresponding to the model
    """
    from nautilus.api.util import summarize_mutation_io
    # compute the appropriate name for the object
    object_type_name = get_model_string(model)

    # return a mutation io object
    return summarize_mutation_io(
        name=object_type_name,
        type=_summarize_object_type(model),
        required=False
    )