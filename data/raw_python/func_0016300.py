def askForFolder(parent, msg = None):
    '''
    Asks for a folder, opening the corresponding dialog with the last path that was selected
    when this same function was invoked from the calling method

    :param parent: The parent window
    :param msg: The message to use for the dialog title
    '''
    msg = msg or 'Select folder'
    caller = _callerName().split(".")
    name = "/".join([LAST_PATH, caller[-1]])
    namespace = caller[0]
    path = pluginSetting(name, namespace)
    folder =  QtWidgets.QFileDialog.getExistingDirectory(parent, msg, path)
    if folder:
        setPluginSetting(name, folder, namespace)
    return folder