def property_get( prop, instance, **kwargs ):
    """Wrapper for property reads which auto-dereferences Refs if required.

    prop
        A Ref (which gets dereferenced and returned) or any other value (which gets returned).

    instance
        The context object used to dereference the Ref.
    """
    if isinstance( prop, Ref ):
        return prop.get( instance, **kwargs )
    return prop