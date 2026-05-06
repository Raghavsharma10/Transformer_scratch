def fieldnames(self):
        ''' return fieldnames
        '''
        if not self._fields:
            if self._item_class is not None:
                for m in inspect.getmembers(self._item_class):
                    if m[0] == 'fields' and isinstance(m[1], dict):
                        self._fields = m[1]
                if not self._fields:
                    raise RuntimeError('Unknown item type, no fields: %s' % self._item_class)
            else:
                raise RuntimeError('Item class is not defined, %s' % self._item_class)
        return self._fields.keys()