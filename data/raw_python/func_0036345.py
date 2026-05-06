def parse_port(port_obj, owner):
    '''Create a port object of the correct type.

    The correct port object type is chosen based on the port.port_type
    property of port_obj.

    @param port_obj The CORBA PortService object to wrap.
    @param owner The owner of this port. Should be a Component object or None.
    @return The created port object.

    '''
    profile = port_obj.get_port_profile()
    props = utils.nvlist_to_dict(profile.properties)
    if props['port.port_type'] == 'DataInPort':
        return DataInPort(port_obj, owner)
    elif props['port.port_type'] == 'DataOutPort':
        return DataOutPort(port_obj, owner)
    elif props['port.port_type'] == 'CorbaPort':
        return CorbaPort(port_obj, owner)
    else:
        return Port(port_obj, owner)