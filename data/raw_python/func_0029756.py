def valid_vlan_id(vlan_id, extended=True):
    """Validates a VLAN ID.

    Args:
        vlan_id (integer): VLAN ID to validate.  If passed as ``str``, it will
            be cast to ``int``.
        extended (bool): If the VLAN ID range should be considered extended
            for Virtual Fabrics.

    Returns:
        bool: ``True`` if it is a valid VLAN ID.  ``False`` if not.

    Raises:
        None

    Examples:
        >>> import pynos.utilities
        >>> vlan = '565'
        >>> pynos.utilities.valid_vlan_id(vlan)
        True
        >>> extended = False
        >>> vlan = '6789'
        >>> pynos.utilities.valid_vlan_id(vlan, extended=extended)
        False
        >>> pynos.utilities.valid_vlan_id(vlan)
        True
    """
    minimum_vlan_id = 1
    maximum_vlan_id = 4095
    if extended:
        maximum_vlan_id = 8191
    return minimum_vlan_id <= int(vlan_id) <= maximum_vlan_id