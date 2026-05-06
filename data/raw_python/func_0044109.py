def get_all_aminames(i_info):
    """Get Image_Name for each instance in i_info.

    Args:
        i_info (dict): information on instances and details.
    Returns:
        i_info (dict): i_info is returned with the aminame
                       added for each instance.

    """
    for i in i_info:
        try:
            # pylint: disable=maybe-no-member
            i_info[i]['aminame'] = EC2R.Image(i_info[i]['ami']).name
        except AttributeError:
            i_info[i]['aminame'] = "Unknown"
    return i_info