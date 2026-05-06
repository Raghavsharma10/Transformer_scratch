def del_object(self, obj):
        """Debug deletes obj of obj[_type] with id of obj['_id']"""
        if obj['_index'] is None or obj['_index'] == "":
            raise Exception("Invalid Object")
        if obj['_id'] is None or obj['_id'] == "":
            raise Exception("Invalid Object")
        if obj['_type'] is None or obj['_type'] == "":
            raise Exception("Invalid Object")
        self.connect_es()
        self.es.delete(index=obj['_index'],
                       id=obj['_id'],
                       doc_type=obj['_type'])