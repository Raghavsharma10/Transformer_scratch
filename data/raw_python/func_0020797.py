def putBlock(self, blockcontent, migration=False):
        """
        Insert the data in sereral steps and commit when each step finishes or rollback if there is a problem.
        """
        #YG
        try:
            #1 insert configuration
            self.logger.debug("insert configuration")
            configList = self.insertOutputModuleConfig(
                            blockcontent['dataset_conf_list'], migration)
            #2 insert dataset
            self.logger.debug("insert dataset")
            datasetId = self.insertDataset(blockcontent, configList, migration)
            #3 insert block & files
            self.logger.debug("insert block & files.")
            self.insertBlockFile(blockcontent, datasetId, migration)
        except KeyError as ex:
            dbsExceptionHandler("dbsException-invalid-input2", "DBSBlockInsert/putBlock: \
                KeyError exception: %s. " %ex.args[0], self.logger.exception, 
	        "DBSBlockInsert/putBlock: KeyError exception: %s. " %ex.args[0]	)
        except Exception as ex:
            raise