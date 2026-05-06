def object_attributes( thing, all=False ) :
    '''
    Return a sorted list of names defined by thing that are not also names in
    a standard object, except include __doc__.
    '''
    standard_names = set( dir( object() ) )
    things_names = set( dir( thing ) )
    if not all :
        things_names -= standard_names
        things_names |= set( ['__doc__'] )
    return sorted( things_names )