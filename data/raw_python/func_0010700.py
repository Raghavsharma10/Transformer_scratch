def decode_pc11_message(raw_string):
    """Decode PC11 message, which usually contains DX Spots"""

    data = {}
    spot = raw_string.split("^")
    data[const.FREQUENCY] = float(spot[1])
    data[const.DX] = spot[2]
    data[const.TIME] = datetime.fromtimestamp(mktime(strptime(spot[3]+" "+spot[4][:-1], "%d-%b-%Y %H%M")))
    data[const.COMMENT] = spot[5]
    data[const.SPOTTER] = spot[6]
    data["node"] = spot[7]
    data["raw_spot"] = raw_string
    return data