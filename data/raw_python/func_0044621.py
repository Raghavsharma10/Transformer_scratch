def gather_data(options):
    """Get Data specific for command selected.

    Create ec2 specific query and output title based on
    options specified, retrieves the raw response data
    from aws, then processes it into the i_info dict,
    which is used throughout this module.

    Args:
        options (object): contains args and data from parser,
                          that has been adjusted by the command
                          specific functions as appropriate.
    Returns:
        i_info (dict): information on instances and details.
        param_str (str): the title to display before the list.

    """
    (qry_string, param_str) = qry_create(options)
    qry_results = awsc.get_inst_info(qry_string)
    i_info = process_results(qry_results)
    return (i_info, param_str)