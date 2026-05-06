def crud_mutation_name(action, model):
    """
        This function returns the name of a mutation that performs the specified
        crud action on the given model service
    """
    model_string = get_model_string(model)
    # make sure the mutation name is correctly camelcases
    model_string = model_string[0].upper() + model_string[1:]

    # return the mutation name
    return "{}{}".format(action, model_string)