def list_files(start_path):
    """tree unix command replacement."""
    s = u'\n'
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        s += u'{}{}/\n'.format(indent, os.path.basename(root))
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            s += u'{}{}\n'.format(sub_indent, f)
    return s