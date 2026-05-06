def DeleteFeatures(self, objectIds=None, where=None, geometry=None,
                       inSR=None, spatialRel=None):
        """This operation deletes features in a feature layer or table (POST
           only). The delete features operation is performed on a feature
           service layer resource. The result of this operation is an array
           of edit results. Each edit result identifies a single feature and
           indicates if the edit were successful or not. If not, it also
           includes an error code and an error description."""
        gt = geometry.__geometry_type__
        if sr is None:
            sr = geometry.spatialReference.wkid
        geo_json = json.dumps(Geometry._json_struct_without_sr)
        return self._get_subfolder("./deleteFeatures", JsonPostResult, {
                                                    'objectIds': objectIds,
                                                    'where': where,
                                                    'geometry': geo_json,
                                                    'geometryType':
                                                            geometryType,
                                                    'inSR': inSR,
                                                    'spatialRel': spatialRel
                                    })