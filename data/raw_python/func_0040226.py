def cluster_files(file_dict):
    '''takes output from :meth:`scan_dir` and organizes into lists of files with the same tags

    returns a dictionary where values are a tuple of the unique tag combination and values contain
    another dictionary with the keys ``info`` containing the original tag dict and ``files`` containing
    a list of files that match'''
    return_dict = {}
    for filename in file_dict:
        info_dict = dict(file_dict[filename])
        if 'md5' in info_dict:
            del(info_dict['md5'])
        dict_key = tuple(sorted([file_dict[filename][x] for x in info_dict]))
        if dict_key not in return_dict:
            return_dict[dict_key] = {'info':info_dict,'files':[]}
        return_dict[dict_key]['files'].append(filename)
    return return_dict