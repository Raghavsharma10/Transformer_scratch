def _make_item(self, item):
        ''' make Item class
        '''
        for field in self._item_class.fields:
            if (field in item) and ('dblite_serializer' in self._item_class.fields[field]):
                serializer = self._item_class.fields[field]['dblite_serializer']
                item[field] = serializer.loads(item[field])
        return self._item_class(item)