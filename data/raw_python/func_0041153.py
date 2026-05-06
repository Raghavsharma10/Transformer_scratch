def do_check_pep8(files, status):
    """
    Run the python pep8 tool against the filst of supplied files.
    Append any linting errors to the returned status list

    Args:
        files (str): list of files to run pep8 against
        status (list): list of pre-receive check failures to eventually print
                       to the user

    Returns:
       status list of current pre-redeive check failures. Might be an empty
       list.
    """
    for file_name in files:

        args = ['flake8', '--max-line-length=120', '{0}'.format(file_name)]
        output = run(*args)

        if output:
            status.append("Python PEP8/Flake8: {0}: {1}".format(file_name,
                                                                output))

    return status