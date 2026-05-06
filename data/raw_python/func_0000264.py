def get_lib2to3_fixers():
    '''returns a list of all fixers found in the lib2to3 library'''
    fixers = []
    fixer_dirname = fixer_dir.__path__[0]
    for name in sorted(os.listdir(fixer_dirname)):
        if name.startswith("fix_") and name.endswith(".py"):
            fixers.append("lib2to3.fixes." + name[:-3])
    return fixers