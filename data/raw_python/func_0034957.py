def print_object_attributes( thing, heading=None, file=None ):
    '''
    Print the attribute names in thing vertically
    '''
    if heading : print( '==', heading, '==', file=file )
    print( '\n'.join( object_attributes( thing ) ), file=file )