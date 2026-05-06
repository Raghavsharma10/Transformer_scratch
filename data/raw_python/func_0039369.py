def text(path, operation, content):
    """
    Perform changes on text files

    :type path: string
    :param path: The path to perform the action on

    :type operation: string
    :param operation: The operation to use on the file

    :type content: string
    :param content: The content to use with the operation
    """

    # If the operation is "write"
    if operation.lower() == 'write':
        # Open the file as "fh"
        with open(path, 'w') as fh:
            # Write to the file
            fh.write(content)

    # If the operation is "append"
    elif operation.lower() == 'append':
        # Open the file as "fh"
        with open(path, 'a') as fh:
            # Write to the file
            fh.write(content)

    # Raise a warning
    raise ValueError("Invalid operation provided")