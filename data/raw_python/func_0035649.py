def get_long():
    """
    Generates a random long. The length of said long varies by platform.
    """
    # The C long type to populate.
    pbRandomData = c_ulong()
    # Determine the byte size of this machine's long type.
    size_of_long = wintypes.DWORD(sizeof(pbRandomData))
    # Used to keep track of status. 1 = success, 0 = error.
    ok = c_int()
    # Provider?
    hProv = c_ulong()

    ok = windll.Advapi32.CryptAcquireContextA(byref(hProv), None, None, PROV_RSA_FULL, 0)
    ok = windll.Advapi32.CryptGenRandom(hProv, size_of_long, byref(pbRandomData))

    return pbRandomData.value