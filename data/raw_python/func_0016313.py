def loadLayer(filename, name = None, provider=None):
    '''
    Tries to load a layer from the given file

    :param filename: the path to the file to load.

    :param name: the name to use for adding the layer to the current project.
    If not passed or None, it will use the filename basename
    '''
    name = name or os.path.splitext(os.path.basename(filename))[0]
    if provider != 'gdal': # QGIS3 crashes if opening a raster as vector ... this needs further investigations
        qgslayer = QgsVectorLayer(filename, name, provider or "ogr")
    if provider == 'gdal' or not qgslayer.isValid():
        qgslayer = QgsRasterLayer(filename, name, provider or "gdal")
        if not qgslayer.isValid():
            raise RuntimeError('Could not load layer: ' + unicode(filename))

    return qgslayer