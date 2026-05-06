def delete_tags(self, archive_name, tags):
        '''
        Delete tags from an archive

        Parameters
        ----------
        archive_name:s tr
            Name of archive

        tags: list or tuple of strings
            tags to delete from the archive

        '''
        updated_tag_list = list(self._get_tags(archive_name))
        for tag in tags:
            if tag in updated_tag_list:
                updated_tag_list.remove(tag)

        self._set_tags(archive_name, updated_tag_list)