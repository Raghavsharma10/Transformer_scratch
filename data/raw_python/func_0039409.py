def register_workflow(connection, domain, workflow):
    """Register a workflow type.

    Return False if this workflow already registered (and True otherwise).
    """
    args = get_workflow_registration_parameter(workflow)

    try:
        connection.register_workflow_type(domain=domain, **args)
    except ClientError as err:
        if err.response['Error']['Code'] == 'TypeAlreadyExistsFault':
            return False  # Ignore this error
        raise

    return True