def tfclasses():
    """
    A mapping of mimetypes to every class for reading data files.
    """
    # automatically find any subclasses of TraceFile in the same
    # directory as me
    classes = {}
    mydir = op.dirname(op.abspath(inspect.getfile(get_mimetype)))
    tfcls = {"<class 'aston.tracefile.TraceFile'>",
             "<class 'aston.tracefile.ScanListFile'>"}
    for filename in glob(op.join(mydir, '*.py')):
        name = op.splitext(op.basename(filename))[0]
        module = import_module('aston.tracefile.' + name)
        for clsname in dir(module):
            cls = getattr(module, clsname)
            if hasattr(cls, '__base__'):
                if str(cls.__base__) in tfcls:
                    classes[cls.mime] = cls
    return classes