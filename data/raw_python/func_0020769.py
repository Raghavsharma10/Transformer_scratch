def listBlockOrigin(self, origin_site_name="",  dataset="", block_name=""):
        """
        API to list blocks first generated in origin_site_name.

        :param origin_site_name: Origin Site Name (Optional, No wildcards)
        :type origin_site_name: str
        :param dataset: dataset ( No wildcards, either dataset or block name needed)
        :type dataset: str
        :param block_name:
        :type block_name: str
        :returns: List of dictionaries containing the following keys (create_by, creation_date, open_for_writing, last_modified_by, dataset, block_name, file_count, origin_site_name, last_modification_date, block_size)
        :rtype: list of dicts

        """
        try:
            return self.dbsBlock.listBlocksOrigin(origin_site_name, dataset, block_name)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listBlocks. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'],
                                self.logger.exception, sError)