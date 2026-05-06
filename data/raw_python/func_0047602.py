def get_object_map(self):
        """stub"""
        obj_map = self._payload.get_object_map()
        obj_map.update({'url': self.get_url()})
        # obj_map['recordTypeIds'].append(str(FILESYSTEM_ASSET_CONTENT_RECORD_TYPE))
        return obj_map