def determine_inst(i_info, param_str, command):
    """Determine the instance-id of the target instance.

    Inspect the number of instance-ids collected and take the
    appropriate action: exit if no ids, return if single id,
    and call user_picklist function if multiple ids exist.

    Args:
        i_info (dict): information and details for instances.
        param_str (str): the title to display in the listing.
        command (str): command specified on the command line.
    Returns:
        tar_inst (str): the AWS instance-id of the target.
    Raises:
        SystemExit: if no instances are match parameters specified.

    """
    qty_instances = len(i_info)
    if not qty_instances:
        print("No instances found with parameters: {}".format(param_str))
        sys.exit(1)

    if qty_instances > 1:
        print("{} instances match these parameters:".format(qty_instances))
        tar_idx = user_picklist(i_info, command)

    else:
        tar_idx = 0
    tar_inst = i_info[tar_idx]['id']
    print("{0}{3}ing{1} instance id {2}{4}{1}".
          format(C_STAT[command], C_NORM, C_TI, command, tar_inst))
    return (tar_inst, tar_idx)