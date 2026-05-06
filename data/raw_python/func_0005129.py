def format_obj_name(obj, delim="<>"):
    """ Formats the object name in a pretty way

        @obj: any python object
        @delim: the characters to wrap a parent object name in

        -> #str formatted name
        ..
            from vital.debug import format_obj_name

            format_obj_name(vital.debug.Timer)
            # -> 'Timer<vital.debug>'

            format_obj_name(vital.debug)
            # -> 'debug<vital>'

            format_obj_name(vital.debug.Timer.time)
            # -> 'time<vital.debug.Timer>'
        ..
    """
    pname = ""
    parent_name = get_parent_name(obj)
    if parent_name:
        pname = "{}{}{}".format(delim[0], get_parent_name(obj), delim[1])
    return "{}{}".format(get_obj_name(obj), pname)