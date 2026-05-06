def user_entry(entry_int, num_inst, command):
    """Validate user entry and returns index and validity flag.

    Processes the user entry and take the appropriate action: abort
    if '0' entered, set validity flag and index is valid entry, else
    return invalid index and the still unset validity flag.

    Args:
        entry_int (int): a number entered or 999 if a non-int was entered.
        num_inst (int): the largest valid number that can be entered.
        command (str): program command to display in prompt.
    Returns:
        entry_idx(int): the dictionary index number of the targeted instance
        valid_entry (bool): specifies if entry_idx is valid.
    Raises:
        SystemExit: if the user enters 0 when they are choosing from the
                    list it triggers the "abort" option offered to the user.

    """
    valid_entry = False
    if not entry_int:
        print("{}aborting{} - {} instance\n".
              format(C_ERR, C_NORM, command))
        sys.exit()
    elif entry_int >= 1 and entry_int <= num_inst:
        entry_idx = entry_int - 1
        valid_entry = True
    else:
        print("{}Invalid entry:{} enter a number between 1"
              " and {}.".format(C_ERR, C_NORM, num_inst))
        entry_idx = entry_int
    return (entry_idx, valid_entry)