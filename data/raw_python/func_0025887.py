def journals(self):
        """
        Retrieve journals attribute for this very Issue
        """
        try:
            target = self._item_path
            json_data = self._redmine.get(target % str(self.id),
                                          parms={'include': 'journals'})
            data = self._redmine.unwrap_json(None, json_data)
            journals = [Journal(redmine=self._redmine,
                                data=journal,
                                type='issue_journal')
                        for journal in data['issue']['journals']]

            return journals

        except Exception:
            return []