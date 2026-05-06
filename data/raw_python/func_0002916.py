def save_dir(key, dir_path, *refs):
    """Convert the given parameters to a special JSON object.

    JSON object is of the form:
    { key: {"dir": dir_path}}, or
    { key: {"dir": dir_path, "refs": [refs[0], refs[1], ... ]}}

    """
    if not os.path.isdir(dir_path):
        return error(
            "Output '{}' set to a missing directory: '{}'.".format(key, dir_path)
        )

    result = {key: {"dir": dir_path}}

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
        result[key]["refs"] = refs

    return json.dumps(result)