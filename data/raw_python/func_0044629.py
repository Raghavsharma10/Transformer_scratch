def user_picklist(i_info, command):
    """Display list of instances matching args and ask user to select target.

    Instance list displayed and user asked to enter the number corresponding
    to the desired target instance, or '0' to abort.

    Args:
        i_info (dict): information on instances and details.
        command (str): command specified on the command line.
    Returns:
        tar_idx (int): the dictionary index number of the targeted instance.

    """
    valid_entry = False
    awsc.get_all_aminames(i_info)
    list_instances(i_info, "", True)
    msg_txt = ("Enter {0}#{1} of instance to {3} ({0}1{1}-{0}{4}{1})"
               " [{2}0 aborts{1}]: ".format(C_WARN, C_NORM, C_TI,
                                            command, len(i_info)))
    while not valid_entry:
        entry_raw = obtain_input(msg_txt)
        try:
            entry_int = int(entry_raw)
        except ValueError:
            entry_int = 999
        (tar_idx, valid_entry) = user_entry(entry_int, len(i_info), command)
    return tar_idx