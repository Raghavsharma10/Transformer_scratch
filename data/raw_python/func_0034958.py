def print_attr_values( thing, all=False, heading=None, file=None ):
    '''
    Print the attributes of thing which have non-empty values,
    as a vertical list of "name : value". When all=True, print
    all attributes even those with empty values.
    '''
    if heading :
        if isinstance( heading, int ) :
            # request for default heading
            heading = '== {} attributes of {} =='.format(
                            'all' if all else 'non-empty',
                            getattr( thing, '__name__', str(thing) )
            )
        print( heading, file=file )

    for attr in object_attributes( thing, all ):
        attr_value = getattr( thing, attr )
        if attr_value is not None :
            print( attr, ':', attr_value, file=file )
        elif all :
            print( attr, ':' )