def view_property( prop ):
    """Wrapper for attributes of a View class which auto-dereferences Refs.
    
    Equivalent to setting a property on the class with the getter wrapped
    with property_get(), and the setter wrapped with property_set().

    prop
        A string containing the name of the class attribute to wrap.
    """
    def getter( self ):
        return property_get( getattr( self, prop ), self.parent )

    def setter( self, value ):
        return property_set( getattr( self, prop ), self.parent, value )

    return property( getter, setter )