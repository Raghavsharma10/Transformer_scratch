def remove_decorators(src):
    """ Remove decorators from the source code """
    src = src.strip()
    src_lines = src.splitlines()
    multi_line = False
    n_deleted = 0
    for n in range(len(src_lines)):
        line = src_lines[n - n_deleted].strip()
        if (line.startswith('@') and 'Benchmark' in line) or multi_line:
            del src_lines[n - n_deleted]
            n_deleted += 1
            if line.endswith(')'):
                multi_line = False
            else:
                multi_line = True
    setup_src = '\n'.join(src_lines)
    return setup_src