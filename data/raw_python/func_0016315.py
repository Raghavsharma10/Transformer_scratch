def addSettingsMenu(menuName, parentMenuFunction=None):
    '''
    Adds a 'open settings...' menu to the plugin menu.
    This method should be called from the initGui() method of the plugin

    :param menuName: The name of the plugin menu in which the settings menu is to be added
    :param parentMenuFunction: a function from QgisInterface to indicate where to put the container plugin menu.
    If not passed, it uses addPluginToMenu
    '''

    parentMenuFunction = parentMenuFunction or iface.addPluginToMenu
    namespace = _callerName().split(".")[0]
    settingsAction = QAction(
        QgsApplication.getThemeIcon('/mActionOptions.svg'),
        "Plugin Settings...",
        iface.mainWindow())
    settingsAction.setObjectName(namespace + "settings")
    settingsAction.triggered.connect(lambda: openSettingsDialog(namespace))
    parentMenuFunction(menuName, settingsAction)
    global _settingActions
    _settingActions[menuName] = settingsAction