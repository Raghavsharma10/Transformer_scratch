def addHelpMenu(menuName, parentMenuFunction=None):
    '''
    Adds a help menu to the plugin menu.
    This method should be called from the initGui() method of the plugin

    :param menuName: The name of the plugin menu in which the about menu is to be added.
    '''

    parentMenuFunction = parentMenuFunction or iface.addPluginToMenu
    namespace = _callerName().split(".")[0]
    path = "file://{}".format(os.path.join(os.path.dirname(_callerPath()), "docs",  "html", "index.html"))    
    helpAction = QtWidgets.QAction(QgsApplication.getThemeIcon('/mActionHelpContents.svg'), 
                                    "Plugin help...", iface.mainWindow())
    helpAction.setObjectName(namespace + "help")
    helpAction.triggered.connect(lambda: openHelp(path))
    parentMenuFunction(menuName, helpAction)
    global _helpActions
    _helpActions[menuName] = helpAction