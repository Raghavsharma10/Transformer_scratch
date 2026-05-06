def delete(self):
        """need this because the JSONClientValidated cannot deal with the magic identifier"""
        magic_identifier = unquote(self.get_id().identifier)
        orig_identifier = magic_identifier.split('?')[0]
        collection = JSONClientValidated('assessment_authoring',
                                         collection='AssessmentPart',
                                         runtime=self.my_osid_object._runtime)
        collection.delete_one({'_id': ObjectId(orig_identifier)})