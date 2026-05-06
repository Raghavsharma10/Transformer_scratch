def _alias_id(self, primary_id, equivalent_id):
        """Adds the given equivalent_id as an alias for primary_id if possible"""
        pkg_name = primary_id.get_identifier_namespace().split('.')[0]
        obj_name = primary_id.get_identifier_namespace().split('.')[1]
        collection = JSONClientValidated(pkg_name,
                                         collection=obj_name,
                                         runtime=self._runtime)
        collection.find_one({'_id': ObjectId(primary_id.get_identifier())})  # to raise NotFound
        collection = JSONClientValidated('id',
                                         collection=pkg_name + 'Ids',
                                         runtime=self._runtime)
        try:
            result = collection.find_one({'aliasIds': {'$in': [str(equivalent_id)]}})
        except errors.NotFound:
            pass
        else:
            result['aliasIds'].remove(str(equivalent_id))
            collection.save(result)
        try:
            id_map = collection.find_one({'_id': str(primary_id)})
        except errors.NotFound:
            collection.insert_one({'_id': str(primary_id), 'aliasIds': [str(equivalent_id)]})
        else:
            id_map['aliasIds'].append(str(equivalent_id))
            collection.save(id_map)