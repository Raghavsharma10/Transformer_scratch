def qry_create(options):
    """Create query from the args specified and command chosen.

    Creates a query string that incorporates the args in the options
    object, and creates the title for the 'list' function.

    Args:
        options (object): contains args and data from parser
    Returns:
        qry_string (str): the query to be used against the aws ec2 client.
        param_str (str): the title to display before the list.

    """
    qry_string = filt_end = param_str = ""
    filt_st = "Filters=["
    param_str_default = "All"

    if options.id:
        qry_string += "InstanceIds=['%s']" % (options.id)
        param_str += "id: '%s'" % (options.id)
        param_str_default = ""

    if options.instname:
        (qry_string, param_str) = qry_helper(bool(options.id),
                                             qry_string, param_str)
        filt_end = "]"
        param_str_default = ""
        qry_string += filt_st + ("{'Name': 'tag:Name', 'Values': ['%s']}"
                                 % (options.instname))
        param_str += "name: '%s'" % (options.instname)

    if options.inst_state:
        (qry_string, param_str) = qry_helper(bool(options.id),
                                             qry_string, param_str,
                                             bool(options.instname), filt_st)
        qry_string += ("{'Name': 'instance-state-name',"
                       "'Values': ['%s']}" % (options.inst_state))
        param_str += "state: '%s'" % (options.inst_state)
        filt_end = "]"
        param_str_default = ""

    qry_string += filt_end
    param_str += param_str_default
    debg.dprintx("\nQuery String")
    debg.dprintx(qry_string, True)
    debg.dprint("param_str: ", param_str)
    return(qry_string, param_str)