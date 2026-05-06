def listDatasetAccessTypes(self, dataset_access_type=''):
        """
        API to list dataset access types.

        :param dataset_access_type: List that dataset access type (Optional)
        :type dataset_access_type: str
        :returns: List of dictionary containing the following key (dataset_access_type).
        :rtype: List of dicts

        """
        if dataset_access_type:
            dataset_access_type = dataset_access_type.replace("*", "%")
        try:
            return self.dbsDatasetAccessType.listDatasetAccessTypes(dataset_access_type)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listDatasetAccessTypes. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)