def print_tasknames(pkgName, aDir, term_width=80, always=False,
                    hidden=None):
    """ Print a message listing TEAL-enabled tasks available under a
        given installation directory (where pkgName resides).
        If always is True, this will always print when tasks are
        found; otherwise it will only print found tasks when in interactive
        mode.
        The parameter 'hidden' supports a list of input tasknames that should
        not be reported even though they still exist.
    """
    # See if we can bail out early
    if not always:
        # We can't use the sys.ps1 check if in PyRAF since it changes sys
        if 'pyraf' not in sys.modules:
           # sys.ps1 is only defined in interactive mode
           if not hasattr(sys, 'ps1'):
               return # leave here, we're in someone's script

    # Check for tasks
    taskDict = cfgpars.findAllCfgTasksUnderDir(aDir)
    tasks = [x for x in taskDict.values() if len(x) > 0]
    if hidden: # could even account for a single taskname as input here if needed
        for x in hidden:
            if x in tasks: tasks.remove(x)
    # only be verbose if there something found
    if len(tasks) > 0:
        sortedUniqTasks = sorted(set(tasks))
        if len(sortedUniqTasks) == 1:
            tlines = 'The following task in the '+pkgName+\
                     ' package can be run with TEAL:\n'
        else:
            tlines = 'The following tasks in the '+pkgName+\
                     ' package can be run with TEAL:\n'
        tlines += printColsAuto(sortedUniqTasks, term_width=term_width,
                                min_pad=2)
        print(tlines)