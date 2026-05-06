def grep(rootdir, searched):
    "grep -l <searched>"
    to_inspect = []
    for root, dirs, files in os.walk(rootdir):
        for _file in files:
            if _file.endswith('.rst'):
                to_inspect.append(join(root, _file))

    to_check = ((_file, open(_file, 'r').read()) for _file in to_inspect)
    to_check = filter(lambda item: searched in item[1], to_check)
    to_check = map(lambda item: item[0], to_check)
    return to_check