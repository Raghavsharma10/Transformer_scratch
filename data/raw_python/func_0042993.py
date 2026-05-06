def get_file(self, repository_fname):
        """
        Retrieve an existing :class:`meteorpi_model.FileRecord` by its ID

        :param string repository_fname:
            The file ID
        :return:
            A :class:`meteorpi_model.FileRecord` instance, or None if not found
        """
        search = mp.FileRecordSearch(repository_fname=repository_fname)
        b = search_files_sql_builder(search)
        sql = b.get_select_sql(columns='f.uid, o.publicId AS observationId, f.mimeType, '
                                       'f.fileName, s2.name AS semanticType, f.fileTime, '
                                       'f.fileSize, f.fileMD5, l.publicId AS obstory_id, l.name AS obstory_name, '
                                       'f.repositoryFname',
                               skip=0, limit=1, order='f.fileTime DESC')
        files = list(self.generators.file_generator(sql=sql, sql_args=b.sql_args))
        if not files:
            return None
        return files[0]