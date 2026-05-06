def refresh(self):
        '''Refresh this item from data on the server.
        Will save any unsaved data first.'''

        if not self._item_path:
            raise AttributeError('refresh is not available for %s' % self._type)
        if not self.id:
            raise RedmineError('%s did not come from the Redmine server - no link.' % self._type)

        try:
            self.save()
        except:
            pass

        # Mimic the Redmine_Item_Manager.get command
        target = self._item_path % self.id
        json_data = self._redmine.get(target)
        data = self._redmine.unwrap_json(self._type, json_data)
        self._update_data(data=data)