def dirtool(operation, directory):
    """
    Tools For Directories (If Exists, Make And Delete)

    :raises ValueError: Nor a string or a list was provided.
    """
    operation = operation.lower()
    if operation == 'exists':
        return bool(os.path.exists(directory))
    if operation == 'create':
        os.makedirs(directory)
    elif operation == 'delete':
        os.rmdir(directory)
    else:
        raise ValueError('Invalid operation provided.')