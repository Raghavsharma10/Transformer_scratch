def filter_python_files(files):
    "Get all python files from the list of files."
    py_files = []
    for f in files:
        # If we end in .py, or if we don't have an extension and file says that
        # we are a python script, then add us to the list
        extension = os.path.splitext(f)[-1]

        if extension:
            if extension == '.py':
                py_files.append(f)
        elif 'python' in open(f, 'r').readline():
            py_files.append(f)
        elif 'python script' in bash('file {}'.format(f)).value().lower():
            py_files.append(f)

    return py_files