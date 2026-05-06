def dumpBlock(self, block_name):
        """
        API the list all information related with the block_name

        :param block_name: Name of block to be dumped (Required)
        :type block_name: str

        """
        try:
            return self.dbsBlock.dumpBlock(block_name)
        except HTTPError as he:
            raise he
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/dumpBlock. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', ex.message, self.logger.exception, sError)