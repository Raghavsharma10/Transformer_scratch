def wif_to_privkey(wif, compressed=True, net=BC):
    """Convert Wallet Import Format (WIF) to privkey bytes."""
    key = b58decode(wif)

    version, raw, check = key[0:1], key[1:-4], key[-4:]
    assert version == net.wifprefix, "unexpected version byte"

    check_compare = shasha(version + raw).digest()[:4]
    assert check_compare == check

    if compressed:
        raw = raw[:-1]

    return raw