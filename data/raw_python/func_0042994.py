def search_files(self, search):
        """
        Search for :class:`meteorpi_model.FileRecord` entities

        :param search:
            an instance of :class:`meteorpi_model.FileRecordSearch` used to constrain the observations returned from
            the DB
        :return:
            a structure of {count:int total rows of an unrestricted search, observations:list of
            :class:`meteorpi_model.FileRecord`}
        """
        b = search_files_sql_builder(search)
        sql = b.get_select_sql(columns='f.uid, o.publicId AS observationId, f.mimeType, '
                                       'f.fileName, s2.name AS semanticType, f.fileTime, '
                                       'f.fileSize, f.fileMD5, l.publicId AS obstory_id, l.name AS obstory_name, '
                                       'f.repositoryFname',
                               skip=search.skip,
                               limit=search.limit,
                               order='f.fileTime DESC')
        files = list(self.generators.file_generator(sql=sql, sql_args=b.sql_args))
        rows_returned = len(files)
        total_rows = rows_returned + search.skip
        if (rows_returned == search.limit > 0) or (rows_returned == 0 and search.skip > 0):
            self.con.execute(b.get_count_sql(), b.sql_args)
            total_rows = self.con.fetchone()['COUNT(*)']
        return {"count": total_rows,
                "files": files}