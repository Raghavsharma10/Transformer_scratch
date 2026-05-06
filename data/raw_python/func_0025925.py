def new(self, **dict):
        '''Create a new item with the provided dict information.  Returns the new item.'''
        if not self._item_new_path:
            raise AttributeError('new is not available for %s' % self._item_name)

        # Remap various tag to tag_id
        for tag in self._object._remap_to_id:
            self._object._remap_tag_to_tag_id(tag, dict)

        target = self._item_new_path
        payload = json.dumps({self._item_type:dict})
        json_data = self._redmine.post(target, payload)
        data = self._redmine.unwrap_json(self._item_type, json_data)
        data['_source_path'] = target
        return self._objectify(data=data)