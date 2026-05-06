def version():
    '''
    Return the version of the FreeType library being used as a tuple of
    ( major version number, minor version number, patch version number )
    '''
    amajor = FT_Int()
    aminor = FT_Int()
    apatch = FT_Int()
    library = get_handle()
    FT_Library_Version(library, byref(amajor), byref(aminor), byref(apatch))
    return (amajor.value, aminor.value, apatch.value)