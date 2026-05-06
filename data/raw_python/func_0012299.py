def update_mutation_inputs(service):
    """
        Args:
            service : The service being updated by the mutation
        Returns:
            (list) : a list of all of the fields availible for the service. Pk
                is a required field in order to filter the results
    """
    # grab the default list of field summaries
    inputs = _service_mutation_summaries(service)

    # visit each field
    for field in inputs:
        # if we're looking at the id field
        if field['name'] == 'id':
            # make sure its required
            field['required'] = True
        # but no other field
        else:
            # is required
            field['required'] = False

    # return the final list
    return inputs