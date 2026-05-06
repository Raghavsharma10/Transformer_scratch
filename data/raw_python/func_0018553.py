def parse_schema_files(files):
    """
    Parse a list of SQL files and return a dictionary of valid schema
    files where each key is a valid schema file and the corresponding value is
    a tuple containing the source and the target schema.
    """
    f_dict = {}
    for f in files:
        root, ext = os.path.splitext(f)
        if ext != ".sql":
            continue
        vto, vfrom = os.path.split(root)
        vto = os.path.split(vto)[1]
        if is_schema(vto) and is_schema(vfrom):
            f_dict[f] = (vfrom, vto)
    return f_dict