def addAboutMenu(menuName, parentMenuFunction=None):
    '''
    Adds an 'about...' menu to the plugin menu.
    This method should be called from the initGui() method of the plugin

    :param menuName: The name of the plugin menu in which the about menu is to be added
    '''

    parentMenuFunction = parentMenuFunction or iface.addPluginToMenu
    namespace = _callerName().split(".")[0]
    icon = QtGui.QIcon(os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", "help.png"))
    aboutAction = QtWidgets.QAction(icon, "About...", iface.mainWindow())
    aboutAction.setObjectName(namespace + "about")
    aboutAction.triggered.connect(lambda: openAboutDialog(namespace))
    parentMenuFunction(menuName, aboutAction)
    global _aboutActions
    _aboutActions[menuName] = aboutAction