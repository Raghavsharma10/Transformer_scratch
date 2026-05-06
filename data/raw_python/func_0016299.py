def askForFiles(parent, msg = None, isSave = False, allowMultiple = False, exts = "*"):
    '''
    Asks for a file or files, opening the corresponding dialog with the last path that was selected
    when this same function was invoked from the calling method.

    :param parent: The parent window
    :param msg: The message to use for the dialog title
    :param isSave: true if we are asking for file to save
    :param allowMultiple: True if should allow multiple files to be selected. Ignored if isSave == True
    :param exts: Extensions to allow in the file dialog. Can be a single string or a list of them.
    Use "*" to add an option that allows all files to be selected

    :returns: A string with the selected filepath or an array of them, depending on whether allowMultiple is True of False
    '''
    msg = msg or 'Select file'
    caller = _callerName().split(".")
    name = "/".join([LAST_PATH, caller[-1]])
    namespace = caller[0]
    path = pluginSetting(name, namespace)
    f = None
    if not isinstance(exts, list):
        exts = [exts]
    extString = ";; ".join([" %s files (*.%s)" % (e.upper(), e) if e != "*" else "All files (*.*)" for e in exts])
    if allowMultiple:
        ret = QtWidgets.QFileDialog.getOpenFileNames(parent, msg, path, '*.' + extString)
        if ret:
            f = ret[0]
        else:
            f = ret = None
    else:
        if isSave:
            ret = QtWidgets.QFileDialog.getSaveFileName(parent, msg, path, '*.' + extString) or None
            if ret is not None and not ret.endswith(exts[0]):
                ret += "." + exts[0]
        else:
            ret = QtWidgets.QFileDialog.getOpenFileName(parent, msg , path, '*.' + extString) or None
        f = ret

    if f is not None:
        setPluginSetting(name, os.path.dirname(f), namespace)

    return ret