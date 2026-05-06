def getHelpFileAsString(taskname,taskpath):
    """
    This functions will return useful help as a string read from a file
    in the task's installed directory called "<module>.help".

    If no such file can be found, it will simply return an empty string.

    Notes
    -----
    The location of the actual help file will be found under the task's
    installed directory using 'irafutils.rglob' to search all sub-dirs to
    find the file. This allows the help file to be either in the tasks
    installed directory or in any sub-directory, such as a "help/" directory.

    Parameters
    ----------
    taskname: string
        Value of `__taskname__` for a module/task

    taskpath: string
        Value of `__file__` for an installed module which defines the task

    Returns
    -------
    helpString: string
        multi-line string read from the file '<taskname>.help'

    """
    #get the local library directory where the code is stored
    pathsplit=os.path.split(taskpath) # taskpath should be task's __file__
    if taskname.find('.') > -1: # if taskname is given as package.taskname...
        helpname=taskname.split(".")[1]    # taskname should be __taskname__ from task's module
    else:
        helpname = taskname
    localdir = pathsplit[0]
    if localdir == '':
       localdir = '.'
    helpfile=rglob(localdir,helpname+".help")[0]

    if os.access(helpfile,os.R_OK):
        fh=open(helpfile,'r')
        ss=fh.readlines()
        fh.close()
        helpString=""
        for line in ss:
            helpString+=line
    else:
        helpString= ''

    return helpString