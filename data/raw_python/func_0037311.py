def add_tags(self, *tags):
        '''
        Set tags for a given archive
        '''
        normed_tags = self.api.manager._normalize_tags(tags)
        self.api.manager.add_tags(self.archive_name, normed_tags)