def _argsdicts( args, mydict ):
    """A utility generator that pads argument list and dictionary values, will only be called with len( args ) = 0, 1."""
    
    if len( args ) == 0:
        args = None, 
    elif len( args ) == 1:
        args = _totuple( args[0] )
    else:
        raise Exception( "We should have never gotten here." )

    mykeys = list( mydict.keys( ) )
    myvalues = list( map( _totuple, list( mydict.values( ) ) ) )

    maxlength = max( list( map( len, [ args ] + myvalues ) ) )

    for i in range( maxlength ):
        thisdict = { }
        for key, value in zip( mykeys, myvalues ):
            try:
                thisdict[ key ] = value[i]
            except IndexError:
                thisdict[ key ] = value[-1]
        try:
            thisarg = args[i]
        except IndexError:
            thisarg = args[-1]

        yield thisarg, thisdict