def list_instances(i_info, param_str, numbered=False):
    """Display a list of all instances and their details.

    Iterates through all the instances in the dict, and displays
    information for each instance.

    Args:
        i_info (dict): information on instances and details.
        param_str (str): the title to display before the list.
        numbered (bool): optional - indicates wheter the list should be
                         displayed with numbers before each instance.
                         This is used when called from user_picklist.

    """
    print(param_str)

    for i in i_info:
        if numbered:
            print("Instance {}#{}{}".format(C_WARN, i + 1, C_NORM))

        print("  {6}Name: {1}{3:<22}{1}ID: {0}{4:<20}{1:<18}Status: {2}{5}{1}".
              format(C_TI, C_NORM, C_STAT[i_info[i]['state']],
                     i_info[i]['tag']['Name'], i_info[i]['id'],
                     i_info[i]['state'], C_HEAD2))
        print("  AMI: {0}{2:<23}{1}AMI Name: {0}{3:.41}{1}".
              format(C_TI, C_NORM, i_info[i]['ami'], i_info[i]['aminame']))
        list_tags(i_info[i]['tag'])
    debg.dprintx("All Data")
    debg.dprintx(i_info, True)