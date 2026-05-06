def QueryLayer(self, text=None, Geometry=None, inSR=None, 
                   spatialRel='esriSpatialRelIntersects', where=None,
                   outFields=None, returnGeometry=None, outSR=None,
                   objectIds=None, time=None, maxAllowableOffset=None,
                   returnIdsOnly=None):
        """The query operation is performed on a layer resource. The result
           of this operation is a resultset resource. This resource provides
           information about query results including the values for the fields
           requested by the user. If you request geometry information, the
           geometry of each result is also returned in the resultset.

           B{Spatial Relation Options:}
             - esriSpatialRelIntersects
             - esriSpatialRelContains
             - esriSpatialRelCrosses
             - esriSpatialRelEnvelopeIntersects
             - esriSpatialRelIndexIntersects
             - esriSpatialRelOverlaps
             - esriSpatialRelTouches
             - esriSpatialRelWithin"""
        if not inSR:
            if Geometry:
                inSR = Geometry.spatialReference
        out = self._get_subfolder("./query", JsonResult, {
                                               'text': text,
                                               'geometry': geometry,
                                               'inSR': inSR,
                                               'spatialRel': spatialRel,
                                               'where': where,
                                               'outFields': outFields,
                                               'returnGeometry': 
                                                    returnGeometry,
                                               'outSR': outSR,
                                               'objectIds': objectIds,
                                               'time': 
                                                    utils.pythonvaluetotime(
                                                        time),
                                               'maxAllowableOffset':
                                                    maxAllowableOffset,
                                               'returnIdsOnly':
                                                    returnIdsOnly
                                                })
        return gptypes.GPFeatureRecordSetLayer.fromJson(out._json_struct)