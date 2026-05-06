def delete_mutation_inputs(service):
    """
        Args:
            service : The service being deleted by the mutation
        Returns:
            ([str]):  the only input for delete is the pk of the service.
    """
    from nautilus.api.util import summarize_mutation_io

    # the only input for delete events is the pk of the service record
    return [summarize_mutation_io(name='pk', type='ID', required=True)]