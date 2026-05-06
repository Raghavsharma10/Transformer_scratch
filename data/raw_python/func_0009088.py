def extract_guts(image_path,
                 tar,
                 file_filter=None,
                 tag_root=True,
                 include_sizes=True):

    '''extract the file guts from an in memory tarfile. The file is not closed.
       This should not be done for large images.
    '''
    if file_filter is None:
        file_filter = get_level('IDENTICAL')

    results = dict()
    digest = dict()
    allfiles = []

    if tag_root:
        roots = dict()

    if include_sizes: 
        sizes = dict()

    for member in tar:
        member_name = member.name.replace('.','',1)
        allfiles.append(member_name)
        included = False
        if member.isdir() or member.issym():
            continue
        elif assess_content(member,file_filter):
            digest[member_name] = extract_content(image_path, member.name, return_hash=True)
            included = True
        elif include_file(member,file_filter):
            hasher = hashlib.md5()
            buf = member.tobuf()
            hasher.update(buf)
            digest[member_name] = hasher.hexdigest()
            included = True
        if included:
            if include_sizes:
                sizes[member_name] = member.size
            if tag_root:
                roots[member_name] = is_root_owned(member)

    results['all'] = allfiles
    results['hashes'] = digest
    if include_sizes:
        results['sizes'] = sizes
    if tag_root:
        results['root_owned'] = roots
    return results