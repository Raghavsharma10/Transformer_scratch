def custom_display(lhs, rhs):
    """
    lhs: left hand side
    rhs: right hand side

    This function serves to inject the string for the left hand side
    of an assignment
    """

    # This code is mainly copied from IPython/display.py
    # (IPython version 2.3.0)
    kwargs = {}
    raw = kwargs.get('raw', False)
    include = kwargs.get('include')
    exclude = kwargs.get('exclude')
    metadata = kwargs.get('metadata')

    from IPython.core.interactiveshell import InteractiveShell
    from IPython.core.displaypub import publish_display_data

    format = InteractiveShell.instance().display_formatter.format
    format_dict, md_dict = format(rhs, include=include, exclude=exclude)

    # example format_dict (for a sympy expression):
    # {u'image/png': '\x89PNG\r\n\x1a\n\x00 ...\x00\x00IEND\xaeB`\x82',
    #  u'text/latex': '$$- 2 \\pi \\sin{\\left (2 \\pi t \\right )}$$',
    # u'text/plain': u'-2\u22c5\u03c0\u22c5sin(2\u22c5\u03c0\u22c5t)'}

    # it is up to IPython which item value is finally used

    # now merge the lhs into the dict:

    if not isinstance(lhs, str):
        raise TypeError('unexpexted Type for lhs object: %s' %type(lhs))

    new_format_dict = {}
    for key, value in list(format_dict.items()):
        if 'text/plain' in key:
            prefix = "{} := ".format(lhs)
            if value.startswith("array") or value.startswith("matrix"):
                value = format_np_array(value, len(prefix))

            new_value = prefix + value
            new_format_dict[key] = new_value

        elif 'text/latex' in key:
            if value.startswith("$$"):
                # this is the expected case
                new_value = r"$$\verb|%s| := %s" % (lhs, value[2:])
                new_format_dict[key] = new_value
            else:
                # this is unexpected but raising an exceptions seems
                # not necessary; handle like plain text (see above)
                new_value = lhs+' := '+value
                new_format_dict[key] = new_value
        else:
            # this happens e.g. for mime-type (i.e. key) 'image/png'
            new_format_dict[key] = value

    # legacy IPython 2.x support
    if IPython.__version__.startswith('2.'):
        publish_display_data('display', new_format_dict, md_dict)
    else:
        # indeed, I dont know with which version the api changed
        # but it does not really matter (for me)
        publish_display_data(data=new_format_dict, metadata=md_dict)