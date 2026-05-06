def run(files, temp_folder, arg=''):
    "Look for pdb.set_trace() commands in python files."
    parser = get_parser()
    args = parser.parse_args(arg.split())

    py_files = filter_python_files(files)
    if args.ignore:
        orig_file_list = original_files(py_files, temp_folder)
        py_files = set(orig_file_list) - set(args.ignore)
        py_files = [temp_folder + f for f in py_files]

    return check_files(py_files).value()