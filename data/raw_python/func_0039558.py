def change_default(
    kls,
    key,
    new_default,
    new_converter=None,
    new_reference_value=None,
):
    """return a new configman Option object that is a copy of an existing one,
    giving the new one a different default value"""
    an_option = kls.get_required_config()[key].copy()
    an_option.default = new_default
    if new_converter:
        an_option.from_string_converter = new_converter
    if new_reference_value:
        an_option.reference_value_from = new_reference_value
    return an_option