def decode_pc23_message(raw_string):
    """ Decode PC23 Message which usually contains WCY """

    data = {}
    wcy = raw_string.split("^")
    data[const.R] = int(wcy[1])
    data[const.expk] = int(wcy[2])
    data[const.CALLSIGN] = wcy[3]
    data[const.A] = wcy[4]
    data[const.SFI] = wcy[5]
    data[const.K] = wcy[6]
    data[const.AURORA] = wcy[7]
    data["node"] = wcy[7]
    data["ip"] = wcy[8]
    data["raw_data"] = raw_string
    return data