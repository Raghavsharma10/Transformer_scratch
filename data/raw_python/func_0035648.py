def get_bytes(num_bytes):
    """
    Returns a random string of num_bytes length.
    """
    # Is this the way to do it?
    #s = c_ubyte()
    # Or this?
    s = create_string_buffer(num_bytes)
    # Used to keep track of status. 1 = success, 0 = error.
    ok = c_int()
    # Provider?
    hProv = c_ulong()

    ok = windll.Advapi32.CryptAcquireContextA(byref(hProv), None, None, PROV_RSA_FULL, 0)
    ok = windll.Advapi32.CryptGenRandom(hProv, wintypes.DWORD(num_bytes), cast(byref(s), POINTER(c_byte)))

    return s.raw