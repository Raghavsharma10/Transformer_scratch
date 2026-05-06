def listReleaseVersions(self, release_version='', dataset='', logical_file_name=''):
        """
        API to list all release versions in DBS

        :param release_version: List only that release version
        :type release_version: str
        :param dataset: List release version of the specified dataset
        :type dataset: str
        :param logical_file_name: List release version of the logical file name
        :type logical_file_name: str
        :returns: List of dictionaries containing following keys (release_version)
        :rtype: list of dicts

        """
        if release_version:
            release_version = release_version.replace("*", "%")
        try:
            return self.dbsReleaseVersion.listReleaseVersions(release_version, dataset, logical_file_name )
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listReleaseVersions. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)