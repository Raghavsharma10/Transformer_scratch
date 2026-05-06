def get_project_templates(path):
    " Get list of installed templates. "

    try:
        return open(op.join(path, TPLNAME)).read().strip().split(',')
    except IOError:
        raise MakesiteException("Invalid makesite-project: %s" % path)