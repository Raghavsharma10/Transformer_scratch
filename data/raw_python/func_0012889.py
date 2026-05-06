def getfieldindex(data, commdct, objkey, fname):
    """given objkey and fieldname, return its index"""
    objindex = data.dtls.index(objkey)
    objcomm = commdct[objindex]
    for i_index, item in enumerate(objcomm):
        try:
            if item['field'] == [fname]:
                break
        except KeyError as err:
            pass
    return i_index