def getSrcBlocks(self, url, dataset="", block=""):
        """
        Need to list all blocks of the dataset and its parents starting from the top
        For now just list the blocks from this dataset.
        Client type call...
        """
        if block:
            params={'block_name':block, 'open_for_writing':0}
        elif dataset:
            params={'dataset':dataset, 'open_for_writing':0}
        else:
            m = 'DBSMigration: Invalid input.  Either block or dataset name has to be provided'
            e = 'DBSMigrate/getSrcBlocks: Invalid input.  Either block or dataset name has to be provided'
            dbsExceptionHandler('dbsException-invalid-input2', m, self.logger.exception, e )

        return cjson.decode(self.callDBSService(url, 'blocks', params, {}))