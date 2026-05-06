def find_dups(file_dict):
    '''takes output from :meth:`scan_dir` and returns list of duplicate files'''
    found_hashes = {}
    for f in file_dict:
        if file_dict[f]['md5'] not in found_hashes:
            found_hashes[file_dict[f]['md5']] = []
        found_hashes[file_dict[f]['md5']].append(f)
    final_hashes = dict(found_hashes)
    for h in found_hashes:
        if len(found_hashes[h])<2:
            del(final_hashes[h])
    return final_hashes.values()