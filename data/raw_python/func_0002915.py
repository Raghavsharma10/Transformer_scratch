def save_file_list(key, *files_refs):
    """Convert the given parameters to a special JSON object.

    Each parameter is a file-refs specification of the form:
    <file-path>:<reference1>,<reference2>, ...,
    where the colon ':' and the list of references are optional.

    JSON object is of the form:
    { key: {"file": file_path}}, or
    { key: {"file": file_path, "refs": [refs[0], refs[1], ... ]}}

    """
    file_list = []
    for file_refs in files_refs:
        if ':' in file_refs:
            try:
                file_name, refs = file_refs.split(':')
            except ValueError as e:
                return error("Only one colon ':' allowed in file-refs specification.")
        else:
            file_name, refs = file_refs, None
        if not os.path.isfile(file_name):
            return error(
                "Output '{}' set to a missing file: '{}'.".format(key, file_name)
            )
        file_obj = {'file': file_name}

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
            file_obj['refs'] = refs

        file_list.append(file_obj)

    return json.dumps({key: file_list})