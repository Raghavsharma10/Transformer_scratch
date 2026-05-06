def get_encryption(cell, emit_version=False):
    """ Gets the encryption type of a network / cell.
    @param string cell
        A network / cell from iwlist scan.

    @return string
        The encryption type of the network.
    """

    enc = ""
    if matching_line(cell, "Encryption key:") == "off":
        enc = "Open"
    else:
        for line in cell:
            matching = match(line,"IE:")
            if matching == None:
                continue

            wpa = match(matching,"WPA")
            if wpa == None:
                continue

            version_matches = VERSION_RGX.search(wpa)
            if len(version_matches.regs) == 1:
                version = version_matches \
                    .group(0) \
                    .lower() \
                    .replace("version", "") \
                    .strip()
                wpa = wpa.replace(version_matches.group(0), "").strip()
                if wpa == "":
                    wpa = "WPA"
                if emit_version:
                    enc = "{0} v.{1}".format(wpa, version)
                else:
                    enc = wpa
                if wpa == "WPA2":
                    return enc
            else:
                enc = wpa
        if enc == "":
            enc = "WEP"
    return enc