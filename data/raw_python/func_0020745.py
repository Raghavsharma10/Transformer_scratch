def listBlocks(self, dataset="", block_name="", data_tier_name="", origin_site_name="",
                   logical_file_name="", run_num=-1, min_cdate=0, max_cdate=0,
                   min_ldate=0, max_ldate=0, cdate=0,  ldate=0, open_for_writing=-1, detail=False):
        """
        dataset, block_name, data_tier_name or logical_file_name must be passed.
        """
        if (not dataset) or re.search("['%','*']", dataset):
            if (not block_name) or re.search("['%','*']", block_name):
                if (not logical_file_name) or re.search("['%','*']", logical_file_name):
                    if not data_tier_name or re.search("['%','*']", data_tier_name):
                        msg = "DBSBlock/listBlock. You must specify at least one parameter(dataset, block_name,\
			       	data_tier_name, logical_file_name) with listBlocks api"
                        dbsExceptionHandler('dbsException-invalid-input2', msg, self.logger.exception, msg)

        if data_tier_name:
            if not (min_cdate and max_cdate) or (max_cdate-min_cdate)>32*24*3600:
                msg = "min_cdate and max_cdate are mandatory parameters. If data_tier_name parameter is used \
                       the maximal time range allowed is 31 days"
                dbsExceptionHandler('dbsException-invalid-input2', msg, self.logger.exception, msg)
            if detail:
                msg = "DBSBlock/listBlock. Detail parameter not allowed togther with data_tier_name"
                dbsExceptionHandler('dbsException-invalid-input2', msg, self.logger.exception, msg)

        with self.dbi.connection() as conn:
            dao = (self.blockbrieflist, self.blocklist)[detail]
            for item in dao.execute(conn, dataset, block_name, data_tier_name, origin_site_name, logical_file_name, run_num,
                                 min_cdate, max_cdate, min_ldate, max_ldate, cdate,  ldate):
                yield item