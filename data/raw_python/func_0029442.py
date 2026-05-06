def _import_parsers():
    """ Lazy imports to prevent circular dependencies between this module and utils """

    global ARCGIS_NODES
    global ARCGIS_ROOTS
    global ArcGISParser

    global FGDC_ROOT
    global FgdcParser

    global ISO_ROOTS
    global IsoParser

    global VALID_ROOTS

    if ARCGIS_NODES is None or ARCGIS_ROOTS is None or ArcGISParser is None:
        from gis_metadata.arcgis_metadata_parser import ARCGIS_NODES
        from gis_metadata.arcgis_metadata_parser import ARCGIS_ROOTS
        from gis_metadata.arcgis_metadata_parser import ArcGISParser

    if FGDC_ROOT is None or FgdcParser is None:
        from gis_metadata.fgdc_metadata_parser import FGDC_ROOT
        from gis_metadata.fgdc_metadata_parser import FgdcParser

    if ISO_ROOTS is None or IsoParser is None:
        from gis_metadata.iso_metadata_parser import ISO_ROOTS
        from gis_metadata.iso_metadata_parser import IsoParser

    if VALID_ROOTS is None:
        VALID_ROOTS = {FGDC_ROOT}.union(ARCGIS_ROOTS + ISO_ROOTS)