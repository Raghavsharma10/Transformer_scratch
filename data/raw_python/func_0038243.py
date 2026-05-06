def filter_ints_based_on_vlan(interfaces, vlan, count=1):
    """ Filter list of interfaces based on VLAN presence or absence criteria.

    :param interfaces: list of interfaces to filter.
    :param vlan: boolean indicating whether to filter interfaces with or without VLAN.
    :param vlan: number of expected VLANs (note that when vlanEnable == False, vlanCount == 1)
    :return: interfaces with VLAN(s) if vlan == True and vlanCount == count else interfaces without
        VLAN(s).

    :todo: add vlanEnable and vlanCount to interface/range/deviceGroup classes.
    """

    filtered_interfaces = []
    for interface in interfaces:
        if interface.obj_type() == 'interface':
            ixn_vlan = interface.get_object_by_type('vlan')
            vlanEnable = is_true(ixn_vlan.get_attribute('vlanEnable'))
            vlanCount = int(ixn_vlan.get_attribute('vlanCount'))
        elif interface.obj_type() == 'range':
            ixn_vlan = interface.get_object_by_type('vlanRange')
            vlanEnable = is_true(ixn_vlan.get_attribute('enabled'))
            vlanCount = len(ixn_vlan.get_objects_by_type('vlanIdInfo'))
        else:
            ixn_vlan = interface.get_object_by_type('ethernet')
            vlanEnable = is_true(ixn_vlan.get_attribute('useVlans'))
            vlanCount = int(ixn_vlan.get_attribute('vlanCount'))
        if not (vlanEnable ^ vlan) and vlanCount == count:
            filtered_interfaces.append(interface)
    return filtered_interfaces