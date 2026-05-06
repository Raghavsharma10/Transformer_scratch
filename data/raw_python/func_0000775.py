def get_es_object_ids(self, objects):
        """ Return IDs of :objects: if they are not IDs already. """
        id_field = self.clean_id_name
        ids = [getattr(obj, id_field, obj) for obj in objects]
        return list(set(str(id_) for id_ in ids))