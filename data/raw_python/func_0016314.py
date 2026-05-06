def loadLayerNoCrsDialog(filename, name=None, provider=None):
    '''
    Tries to load a layer from the given file
    Same as the loadLayer method, but it does not ask for CRS, regardless of current
    configuration in QGIS settings
    '''
    settings = QSettings()
    prjSetting = settings.value('/Projections/defaultBehaviour')
    settings.setValue('/Projections/defaultBehaviour', '')
    # QGIS3:
    prjSetting3 = settings.value('/Projections/defaultBehavior')
    settings.setValue('/Projections/defaultBehavior', '')
    layer = loadLayer(filename, name, provider)
    settings.setValue('/Projections/defaultBehaviour', prjSetting)
    settings.setValue('/Projections/defaultBehavior', prjSetting3)
    return layer