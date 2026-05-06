def printcodelist(thing, to=sys.stdout, heading=None):
    '''
    Write the lines of the codelist string list to the given file, or to
    the default output.

    A little Python 3 problem: if the to-file is in binary mode, we need to
    encode the strings, else a TypeError will be raised. Obvious answer, test
    for 'b' in to.mode? Nope, only "real" file objects have a mode attribute.
    StringIO objects, and the variant StringIO used as default sys.stdout, do
    not have .mode.

    However, all file-like objects that support string output DO have an
    encoding attribute. (StringIO has one that is an empty string, but it
    exists.) So, if hasattr(to,'encoding'), just shove the whole string into
    it. Otherwise, encode the string utf-8 and shove that bytestring into it.
    (See? Python 3 not so hard...)

    '''
    # If we were passed a list, assume that it is a CodeList or
    # a manually-assembled list of code tuples.
    if not isinstance( thing, list ) :
        # Passed something else. Reduce it to a CodeList.
        if isinstance( thing, Code ):
            thing = thing.code
        else :
            # Convert various sources to a code object.
            thing = _get_a_code_object_from( thing )
            try :
                thing = Code.from_code( thing ).code
            except Exception as e:
                raise ValueError('Invalid input to printcodelist')
    # We have a CodeList or equivalent,
    # get the whole disassembly as a string.
    whole_thang = str( thing )
    # if destination not a text file, encode it to bytes
    if not hasattr( to, 'encoding' ) :
        whole_thang = whole_thang.encode( 'UTF-8' )
        if heading : # is not None or empty
            heading = heading.encode( 'UTF-8' )
    # send it on its way
    if heading :
        to.write( '===' + heading + '===\n' )
    to.write( whole_thang )