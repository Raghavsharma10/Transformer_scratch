def fixup_paths():
    """
    Fixup paths added in .pth file that point to the virtualenv
    instead of the system site packages.

    In depth:  .PTH can execute arbitrary code, which might
    manipulate the PATH or sys.path

    :return:
    """
    original_paths = os.environ.get('PATH', "").split(os.path.pathsep)
    original_dirs = set(added_dirs)
    yield

    # Fix PATH environment variable
    current_paths = os.environ.get('PATH', "").split(os.path.pathsep)
    if original_paths != current_paths:
        changed_paths = set(current_paths).difference(set(original_paths))
        # rebuild PATH env var
        fixed_paths = []
        for path in current_paths:
            if path in changed_paths:
                fixed_paths.append(env_t(fix_path(path)))
            else:
                fixed_paths.append(env_t(path))
        os.environ['PATH'] = os.pathsep.join(fixed_paths)

    # Fix added_dirs
    if added_dirs != original_dirs:
        for path in set(added_dirs.difference(original_dirs)):
            fixed_path = fix_path(path)
            if fixed_path != path:
                print("Fix %s >> %s" % (path, fixed_path))
                added_dirs.remove(path)
                added_dirs.add(fixed_path)

                i = sys.path.index(path)  # not efficient... but shouldn't happen often
                sys.path[i] = fixed_path

                if env_t(fixed_path) not in os.environ['PATH']:
                    os.environ['PATH'].append(os.pathsep + env_t(fixed_path))