def ApplyEdits(self, adds=None, updates=None, deletes=None):
        """This operation adds, updates and deletes features to the associated
           feature layer or table in a single call (POST only). The apply edits
           operation is performed on a feature service layer resource. The
           result of this operation are 3 arrays of edit results (for adds,
           updates and deletes respectively). Each edit result identifies a
           single feature and indicates if the edit were successful or not. If
           not, it also includes an error code and an error description."""
        add_str, update_str = None, None
        if adds:
            add_str = ",".join(json.dumps(
                                        feature._json_struct_for_featureset) 
                                    for feature in adds)
        if updates:
            update_str = ",".join(json.dumps(
                                        feature._json_struct_for_featureset) 
                                    for feature in updates)
        return self._get_subfolder("./applyEdits", JsonPostResult,
                                                                 {'adds':
                                                                       add_str,
                                                                  'updates':
                                                                    update_str,
                                                                   'deletes':
                                                                        deletes
                                                                   })