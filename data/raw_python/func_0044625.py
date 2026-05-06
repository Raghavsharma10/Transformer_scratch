def qry_helper(flag_id, qry_string, param_str, flag_filt=False, filt_st=""):
    """Dynamically add syntaxtical elements to query.

    This functions adds syntactical elements to the query string, and
    report title, based on the types and number of items added thus far.

    Args:
        flag_filt (bool): at least one filter item specified.
        qry_string (str): portion of the query constructed thus far.
        param_str (str): the title to display before the list.
        flag_id (bool): optional - instance-id was specified.
        filt_st (str): optional - syntax to add on end if filter specified.
    Returns:
        qry_string (str): the portion of the query that was passed in with
                          the appropriate syntactical elements added.
        param_str (str): the title to display before the list.

    """
    if flag_id or flag_filt:
        qry_string += ", "
        param_str += ", "

    if not flag_filt:
        qry_string += filt_st
    return (qry_string, param_str)