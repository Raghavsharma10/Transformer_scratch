def get_single_fixer(fixname):
    '''return a single fixer found in the lib2to3 library'''
    fixer_dirname = fixer_dir.__path__[0]
    for name in sorted(os.listdir(fixer_dirname)):
        if (name.startswith("fix_") and name.endswith(".py") 
            and fixname == name[4:-3]):
            return "lib2to3.fixes." + name[:-3]