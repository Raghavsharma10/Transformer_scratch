def create_mutation_inputs(service):
    """
        Args:
            service : The service being created by the mutation
        Returns:
            (list) : a list of all of the fields availible for the service,
                with the required ones respected.
    """
    # grab the default list of field summaries
    inputs = _service_mutation_summaries(service)
    # make sure the pk isn't in the list
    inputs.remove([field for field in inputs if field['name'] == 'id'][0])

    # return the final list
    return inputs