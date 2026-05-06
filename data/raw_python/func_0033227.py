def skeleton_path(parts):
    """Gets the path to a skeleton asset"""
    return os.path.join(os.path.dirname(oz.__file__), "skeleton", parts)