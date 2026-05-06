def get_other_props(all_props, reserved_props):
    # type: (Dict, Tuple) -> Optional[Dict]
    """
    Retrieve the non-reserved properties from a dictionary of properties
    @args reserved_props: The set of reserved properties to exclude
    """
    if hasattr(all_props, 'items') and callable(all_props.items):
        return dict([(k,v) for (k,v) in list(all_props.items()) if k not in
                     reserved_props])
    return None