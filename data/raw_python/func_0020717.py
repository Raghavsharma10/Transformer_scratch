def listReleaseVersions(self, release_version="", dataset='', logical_file_name=''):
        """
        List release versions
        """
        if dataset and ('%' in dataset or '*' in dataset):
            dbsExceptionHandler('dbsException-invalid-input',
                " DBSReleaseVersion/listReleaseVersions. No wildcards are" +
                " allowed in dataset.\n.")

        if logical_file_name and ('%' in logical_file_name or '*' in logical_file_name):
            dbsExceptionHandler('dbsException-invalid-input',
                " DBSReleaseVersion/listReleaseVersions. No wildcards are" +
                " allowed in logical_file_name.\n.")

        conn = self.dbi.connection()
        try:
            plist = self.releaseVersion.execute(conn, release_version.upper(), dataset, logical_file_name)
            result = [{}]
            if plist:
                t = []
                for i in plist:
                    for k, v in i.iteritems():
                        t.append(v)
                result[0]['release_version'] = t
            return result
        finally:
            if conn:
                conn.close()