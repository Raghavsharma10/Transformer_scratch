def privkey_to_wif(rawkey, compressed=True, net=BC):
    """Convert privkey bytes to Wallet Import Format (WIF)."""
    # See https://en.bitcoin.it/wiki/Wallet_import_format.
    k = net.wifprefix + rawkey
    if compressed:
        k += b'\x01'

    chksum = shasha(k).digest()[:4]
    key = k + chksum

    b58key = b58encode(key)
    return b58key