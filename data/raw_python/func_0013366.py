def removeVariantSet(self, variantSet):
        """
        Removes the specified variantSet from this repository. This performs
        a cascading removal of all items within this variantSet.
        """
        for variantSetRecord in models.Variantset.select().where(
                        models.Variantset.id == variantSet.getId()):
            variantSetRecord.delete_instance(recursive=True)