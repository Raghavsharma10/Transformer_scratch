def get_provenance_children(self):
        """stub"""
        if self.has_provenance_children():
            collection = JSONClientValidated('assessment',
                                             collection='Item',
                                             runtime=self.my_osid_object._runtime)
            try:
                result = collection.find(
                    {'provenanceId': self.my_osid_object.object_map['id']})
                if result.count() == 0:
                    raise KeyError
            except KeyError:
                # For deprecated mecqbank data
                result = collection.find(
                    {'provenanceItemId': self.my_osid_object.object_map['id']})
            return ItemList(result,
                            runtime=self.my_osid_object._runtime,
                            proxy=self.my_osid_object._proxy)
        raise IllegalState('No provenance children.')