def collect_cases(data_dir):
    """ Find all cases and subcases of a particular run type """
    cases = {}
    for root, dirs, files in os.walk(data_dir):
        if not dirs:
            split_case = os.path.relpath(root, data_dir).split(os.path.sep)
            if split_case[0] not in cases:
                cases[split_case[0]] = []
            cases[split_case[0]].append("-".join(split_case[1:]))
    return cases