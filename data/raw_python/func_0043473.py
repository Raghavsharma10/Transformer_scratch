def gen_template_files(path):
    " Generate relative template pathes. "

    path = path.rstrip(op.sep)
    for root, _, files in walk(path):
        for f in filter(lambda x: not x in (TPLNAME, CFGNAME), files):
            yield op.relpath(op.join(root, f), path)