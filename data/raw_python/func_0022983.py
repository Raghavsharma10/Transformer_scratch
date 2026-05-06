def get_handle():
    '''
    Get unique FT_Library handle
    '''
    global __handle__
    if not __handle__:
        __handle__ = FT_Library()
        error = FT_Init_FreeType(byref(__handle__))
        if error:
            raise RuntimeError(hex(error))
    return __handle__