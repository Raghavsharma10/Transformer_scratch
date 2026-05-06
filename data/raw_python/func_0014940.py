def runInfo(prog=None,vers=None,date=None,user=None,dir=None,args=None):
    r"""Create a short info string detailing how a program was invoked. This is
    meant to be added to a history comment field of a data file were it is
    important to keep track of what programs modified it and how.

    !!!:`args` should be a **``list``** not a ``str``."""

    return "%(prog)s %(vers)s;" \
           " run %(date)s by %(usr)s in %(dir)s with: %(args)s'n" % \
           mkDict(prog=prog or sys.argv[0],
                  vers=vers or magicGlobals().get("__version__", ""),
                  date=date or isoDateTimeStr(),
                  usr=user or getpass.getuser(),
                  dir=dir or os.getcwd(),
                  args=" ".join(args or sys.argv))