def file_generator(self, sql, sql_args):
        """Generator for FileRecord

        :param sql:
            A SQL statement which must return rows describing files.
        :param sql_args:
            Any variables required to populate the query provided in 'sql'
        :return:
            A generator which produces FileRecord instances from the supplied SQL, closing any opened cursors on
            completion.
        """

        self.con.execute(sql, sql_args)
        results = self.con.fetchall()
        output = []
        for result in results:
            file_record = mp.FileRecord(obstory_id=result['obstory_id'], obstory_name=result['obstory_name'],
                                        observation_id=result['observationId'],
                                        repository_fname=result['repositoryFname'],
                                        file_time=result['fileTime'], file_size=result['fileSize'],
                                        file_name=result['fileName'], mime_type=result['mimeType'],
                                        file_md5=result['fileMD5'],
                                        semantic_type=result['semanticType'])

            # Look up observation metadata
            sql = """SELECT f.metaKey, stringValue, floatValue
FROM archive_metadata m
INNER JOIN archive_metadataFields f ON m.fieldId=f.uid
WHERE m.fileId=%s
"""
            self.con.execute(sql, (result['uid'],))
            for item in self.con.fetchall():
                value = first_non_null([item['stringValue'], item['floatValue']])
                file_record.meta.append(mp.Meta(item['metaKey'], value))

            output.append(file_record)
        return output