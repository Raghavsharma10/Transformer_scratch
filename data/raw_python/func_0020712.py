def listFileParentsByLumi(self, block_name='', logical_file_name=[]):
        """
        required parameter: block_name
        returns:  [{child_parent_id_list: [(cid1, pid1), (cid2, pid2), ... (cidn, pidn)]}]
        """
        #self.logger.debug("lfn %s, block_name %s" % (logical_file_name, block_name))
        if not block_name:
            dbsExceptionHandler('dbsException-invalid-input', \
                "Child block_name is required for fileparents/listFileParentsByLumi api", self.logger.exception )
        with self.dbi.connection() as conn:
            sqlresult = self.fileparentbylumi.execute(conn, block_name, logical_file_name)
            return [{"child_parent_id_list":sqlresult}]