def set_dump_directory(base=None, sub_dir=None):
    """Create directory for dumping SQL commands."""
    # Set current timestamp
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H-%M-%S')

    # Clean sub_dir
    if sub_dir and '.' in sub_dir:
        sub_dir = sub_dir.rsplit('.', 1)[0]

    # Create a directory to save fail SQL scripts
    # TODO: Replace with function that recursively creates directories until path exists
    if not os.path.exists(base):
        os.mkdir(base)
    dump_dir = os.path.join(base, sub_dir) if sub_dir else base
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    dump_dir = os.path.join(dump_dir, timestamp)
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
        return dump_dir