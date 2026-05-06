def CreateShortcut(Path, Target, Arguments="", StartIn="", Icon=("", 0), Description=""):
    """Create a Windows shortcut:

    Path - As what file should the shortcut be created?
    Target - What command should the desktop use?
    Arguments - What arguments should be supplied to the command?
    StartIn - What folder should the command start in?
    Icon -(filename, index) What icon should be used for the shortcut?
    Description - What description should the shortcut be given?

    eg
    CreateShortcut(
        Path=os.path.join(desktop(), "PythonI.lnk"),
        Target=r"c:\python\python.exe",
        Icon=(r"c:\python\python.exe", 0),
        Description="Python Interpreter"
    )
    """
    lnk = shortcut(Target)
    lnk.arguments = Arguments
    lnk.working_directory = StartIn
    lnk.icon_location = Icon
    lnk.description = Description
    lnk.write(Path)