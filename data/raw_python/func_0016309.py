def addLayerNoCrsDialog(layer, loadInLegend=True):
    '''
    Tries to add a layer from layer object
    Same as the addLayer method, but it does not ask for CRS, regardless of current
    configuration in QGIS settings
    '''
    settings = QSettings()
    prjSetting = settings.value('/Projections/defaultBehaviour')
    settings.setValue('/Projections/defaultBehaviour', '')
    # QGIS3
    prjSetting3 = settings.value('/Projections/defaultBehavior')
    settings.setValue('/Projections/defaultBehavior', '')
    layer = addLayer(layer, loadInLegend)
    settings.setValue('/Projections/defaultBehaviour', prjSetting)
    settings.setValue('/Projections/defaultBehavior', prjSetting3)
    return layer