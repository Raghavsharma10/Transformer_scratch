def dump_element(element):
    """
    Dumps the content of the given ElementBase object to a string

    :param element: An ElementBase object
    :return: A full description of its content
    :raise TypeError: Invalid object
    """
    # Check type
    try:
        assert isinstance(element, sleekxmpp.ElementBase)
    except AssertionError:
        raise TypeError("Not an ElementBase: {0}".format(type(element)))

    # Prepare string
    output = StringIO()
    output.write("ElementBase : {0}\n".format(type(element)))
    output.write("- name......: {0}\n".format(element.name))
    output.write("- namespace.: {0}\n".format(element.namespace))

    output.write("- interfaces:\n")
    for itf in sorted(element.interfaces):
        output.write("\t- {0}: {1}\n".format(itf, element[itf]))

    if element.sub_interfaces:
        output.write("- sub-interfaces:\n")
        for itf in sorted(element.sub_interfaces):
            output.write("\t- {0}: {1}\n".format(itf, element[itf]))

    return output.getvalue()