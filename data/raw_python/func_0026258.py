def list_id(self):
        """ Get or create the list id. """
        list_id = getattr(self, '_list_id', None)

        if list_id is None:
            for l in self.api.lists.all()['lists']:
                if l['name'] == self.list_name:
                    self._list_id = l['id']

            if not getattr(self, '_list_id', None):
                self._list_id = self.api.lists.create(
                    label=self.list_label, name=self.list_name,
                    method='POST')['list_id']

        return self._list_id