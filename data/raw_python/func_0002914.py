def save_file(key, file_path, *refs):
    """Convert the given parameters to a special JSON object.

    JSON object is of the form:
    { key: {"file": file_path}}, or
    { key: {"file": file_path, "refs": [refs[0], refs[1], ... ]}}

    """
    if not os.path.isfile(file_path):
        return error("Output '{}' set to a missing file: '{}'.".format(key, file_path))

    result = {key: {"file": file_path}}

    if refs:
        missing_refs = [
            ref for ref in refs if not (os.path.isfile(ref) or os.path.isdir(ref))
        ]
        if len(missing_refs) > 0:
            return error(
                "Output '{}' set to missing references: '{}'.".format(
                    key, ', '.join(missing_refs)
                )
            )
        result[key]['refs'] = refs

    return json.dumps(result)