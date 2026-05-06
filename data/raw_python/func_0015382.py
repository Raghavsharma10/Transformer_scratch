def addpackage(sys_sitedir, pthfile, known_dirs):
    """
    Wrapper for site.addpackage

    Try and work out which directories are added by
    the .pth and add them to the known_dirs set

    :param sys_sitedir: system site-packages directory
    :param pthfile: path file to add
    :param known_dirs: set of known directories
    """
    with open(join(sys_sitedir, pthfile)) as f:
        for n, line in enumerate(f):
            if line.startswith("#"):
                continue
            line = line.rstrip()
            if line:
                if line.startswith(("import ", "import\t")):
                    exec (line, globals(), locals())
                    continue
                else:
                    p_rel = join(sys_sitedir, line)
                    p_abs = abspath(line)
                    if isdir(p_rel):
                        os.environ['PATH'] += env_t(os.pathsep + p_rel)
                        sys.path.append(p_rel)
                        added_dirs.add(p_rel)
                    elif isdir(p_abs):
                        os.environ['PATH'] += env_t(os.pathsep + p_abs)
                        sys.path.append(p_abs)
                        added_dirs.add(p_abs)

    if isfile(pthfile):
        site.addpackage(sys_sitedir, pthfile, known_dirs)
    else:
        logging.debug("pth file '%s' not found")