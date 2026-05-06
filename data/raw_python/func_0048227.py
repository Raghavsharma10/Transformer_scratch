def add_choice(self, asset_id, name='', identifier=None):
        """stub"""
        if identifier is None:
            identifier = str(ObjectId())
        self.my_osid_object_form._my_map['choices'].append({
            'id': identifier,
            'assetId': str(asset_id),
            'name': name
        })
        return identifier