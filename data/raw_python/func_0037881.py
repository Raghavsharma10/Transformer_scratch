def write_json(data, path, file_name):
    """
    Write out data to a json file.

    Args:
        data: A dictionary representation of the data to write out
        path: The directory to output the file in
        file_name: The name of the file to write out
    """
    if os.path.exists(path) and not os.path.isdir(path):
        return
    elif not os.path.exists(path):
        mkdir_p(path)
    with open(os.path.join(path, file_name), 'w') as f:
        json_tricks.dump(data, f, indent=4, primitives=True, allow_nan=True)