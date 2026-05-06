def save_dir_list(key, *dirs_refs):
    """Convert the given parameters to a special JSON object.

    Each parameter is a dir-refs specification of the form:
    <dir-path>:<reference1>,<reference2>, ...,
    where the colon ':' and the list of references are optional.

    JSON object is of the form:
    { key: {"dir": dir_path}}, or
    { key: {"dir": dir_path, "refs": [refs[0], refs[1], ... ]}}

    """
    dir_list = []
    for dir_refs in dirs_refs:
        if ':' in dir_refs:
            try:
                dir_path, refs = dir_refs.split(':')
            except ValueError as e:
                return error("Only one colon ':' allowed in dir-refs specification.")
        else:
            dir_path, refs = dir_refs, None
        if not os.path.isdir(dir_path):
            return error(
                "Output '{}' set to a missing directory: '{}'.".format(key, dir_path)
            )
        dir_obj = {'dir': dir_path}

        if refs:
            refs = [ref_path.strip() for ref_path in refs.split(',')]
            missing_refs = [
                ref for ref in refs if not (os.path.isfile(ref) or os.path.isdir(ref))
            ]
            if len(missing_refs) > 0:
                return error(
                    "Output '{}' set to missing references: '{}'.".format(
                        key, ', '.join(missing_refs)
                    )
                )
            dir_obj['refs'] = refs

        dir_list.append(dir_obj)

    return json.dumps({key: dir_list})