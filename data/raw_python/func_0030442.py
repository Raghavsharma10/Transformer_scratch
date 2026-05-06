def ensure_dir_exists(path):
    """Given a file, ensure that the path to the file exists"""

    import os

    f_dir = os.path.dirname(path)

    if not os.path.exists(f_dir):
        os.makedirs(f_dir)

    return f_dir