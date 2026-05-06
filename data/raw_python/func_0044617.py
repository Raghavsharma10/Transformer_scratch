def cmd_list(options):
    """Gather data for instances matching args and call display func.

    Args:
        options (object): contains args and data from parser.

    """
    (i_info, param_str) = gather_data(options)
    if i_info:
        awsc.get_all_aminames(i_info)
        param_str = "Instance List - " + param_str + "\n"
        list_instances(i_info, param_str)
    else:
        print("No instances found with parameters: {}".format(param_str))