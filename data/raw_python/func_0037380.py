def get_archive(self, archive_name):
        '''
        Get a data archive given an archive name

        Returns
        -------
        archive_specification : dict
            archive_name: name of the archive to be retrieved
            authority: name of the archive's authority
            archive_path: service path of archive
        '''

        try:
            spec = self._get_archive_spec(archive_name)
            return spec

        except KeyError:
            raise KeyError('Archive "{}" not found'.format(archive_name))