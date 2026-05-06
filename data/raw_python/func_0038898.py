def QueryRelatedRecords(self, objectIds=None, relationshipId=None,
                            outFields=None, definitionExpression=None,
                            returnGeometry=None, outSR=None):
        """The query operation is performed on a feature service layer
           resource. The result of this operation are featuresets grouped by
           source layer / table object IDs. Each featureset contains Feature
           objects including the values for the fields requested by the user.
           For related layers, if you request geometry information, the
           geometry of each feature is also returned in the featureset. For
           related tables, the featureset does not include geometries."""

        out = self._get_subfolder("./queryRelatedRecords", JsonResult, {
                                                        'objectIds':
                                                            objectIds,
                                                        'relationshipId':
                                                            relationshipId,
                                                        'outFields':
                                                            outFields,
                                                        'definitionExpression':
                                                          definitionExpression,
                                                        'returnGeometry':
                                                            returnGeometry,
                                                        'outSR': outSR
                                                })
        return out._json_struct