def scan_dir(dirname,tags=None,md5_hash=False):
    '''scans a directory tree and returns a dictionary with files and key DICOM tags

    return value is a dictionary absolute filenames as keys and with dictionaries of tags/values
    as values

    the param ``tags`` is the list of DICOM tags (given as tuples of hex numbers) that
    will be obtained for each file. If not given,
    the default list is:

    :0008 0021:     Series date
    :0008 0031:     Series time
    :0008 103E:     Series description
    :0008 0080:     Institution name
    :0010 0020:     Patient ID
    :0028 0010:     Image rows
    :0028 0011:     Image columns

    If the param ``md5_hash`` is ``True``, this will also return the MD5 hash of the file. This is useful
    for detecting duplicate files
    '''
    if tags==None:
        tags = [
            (0x0008, 0x0021),
            (0x0008, 0x0031),
            (0x0008, 0x103E),
            (0x0008, 0x0080),
            (0x0010, 0x0020),
            (0x0028, 0x0010),
            (0x0028, 0x0011),
        ]

    return_dict = {}

    for root,dirs,files in os.walk(dirname):
        for filename in files:
            fullname = os.path.join(root,filename)
            if is_dicom(fullname):
                return_dict[fullname] = info_for_tags(fullname,tags)
                if md5_hash:
                    return_dict[fullname]['md5'] = nl.hash(fullname)
    return return_dict