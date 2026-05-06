def valid_as_v2_0(voevent):
    """Tests if a voevent conforms to the schema.

    Args:
        voevent(:class:`Voevent`): Root node of a VOEvent etree.
    Returns:
        bool: Whether VOEvent is valid
    """
    _return_to_standard_xml(voevent)
    valid_bool = voevent_v2_0_schema.validate(voevent)
    _remove_root_tag_prefix(voevent)
    return valid_bool