def do_check(func, files, status):
    """
    Generic do_check helper method

    Args:
        func (function): Specific function to call
        files (list): list of files to run against
        status (list): list of pre-receive check failures to eventually print
                       to the user

    Returns:
       status list of current pre-redeive check failures. Might be an empty
       list.
    """

    for file_name in files:
        with open(file_name, 'r') as f:
            output = func.parse(f.read(), file_name)

        if output:
            status.append("{0}: {1}".format(file_name, output))

    return status