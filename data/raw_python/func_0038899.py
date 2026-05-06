def UpdateFeatures(self, features):
        """This operation updates features to the associated feature layer or
           table (POST only). The update features operation is performed on a
           feature service layer resource. The result of this operation is an
           array of edit results. Each edit result identifies a single feature
           and indicates if the edit were successful or not. If not, it also
           includes an error code and an error description."""
        fd = {'features': ",".join(json.dumps(
                                        feature._json_struct_for_featureset) 
                                    for feature in features)}
        return self._get_subfolder("./updateFeatures", JsonPostResult, fd)