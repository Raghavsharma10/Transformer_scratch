def read_folder(directory):
    """read text files in directory and returns them as array

    Args:
        directory: where the text files are

    Returns:
        Array of text
    """
    res = []
    for filename in os.listdir(directory):
        with io.open(os.path.join(directory, filename), encoding="utf-8") as f:
            content = f.read()
            res.append(content)
    return res