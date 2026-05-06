def property_set( prop, instance, value, **kwargs ):
    """Wrapper for property writes which auto-deferences Refs.

    prop
        A Ref (which gets dereferenced and the target value set).

    instance
        The context object used to dereference the Ref.

    value
        The value to set the property to.

    Throws AttributeError if prop is not a Ref.
    """

    if isinstance( prop, Ref ):
        return prop.set( instance, value, **kwargs )
    raise AttributeError( "can't change value of constant {} (context: {})".format( prop, instance ) )