def add_tags(self, archive_name, tags):
        '''
        Add tags to an archive

        Parameters
        ----------
        archive_name:s tr
            Name of archive

        tags: list or tuple of strings
            tags to add to the archive

        '''
        updated_tag_list = list(self._get_tags(archive_name))
        for tag in tags:
            if tag not in updated_tag_list:
                updated_tag_list.append(tag)

        self._set_tags(archive_name, updated_tag_list)