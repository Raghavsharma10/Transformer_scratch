def get_provenance_parent(self):
        """stub"""
        if self.has_provenance():
            collection = JSONClientValidated('assessment',
                                             collection='Item',
                                             runtime=self.my_osid_object._runtime)
            result = collection.find_one(
                {'_id': ObjectId(Id(self.get_provenance_id()).get_identifier())})
            return Item(osid_object_map=result,
                        runtime=self.my_osid_object._runtime,
                        proxy=self.my_osid_object._proxy)
        raise IllegalState("Item has no provenance parent.")