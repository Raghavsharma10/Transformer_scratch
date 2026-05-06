def info_for_tags(filename,tags):
    '''return a dictionary for the given ``tags`` in the header of the DICOM file ``filename``

    ``tags`` is expected to be a list of tuples that contains the DICOM address in hex values.

    basically a rewrite of :meth:`info` because it's so slow. This is a lot faster and more reliable'''
    if isinstance(tags,tuple):
        tags = [tags]
    d = pydicom.read_file(filename)
    return_dict = {}
    dicom_info = None
    for k in tags:
        if k in d:
            return_dict[k] = d[k].value
        else:
            # Backup to the old method
            if dicom_info==None:
                dicom_info = info(filename)
            i = dicom_info.addr(k)
            if i:
                return_dict[k] = nl.numberize(i['value'])
    return return_dict