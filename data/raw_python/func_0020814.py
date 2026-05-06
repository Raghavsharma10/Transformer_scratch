def listBlocks(self, **kwargs):
        """
        API to list a block in DBS. At least one of the parameters block_name, dataset, data_tier_name or
        logical_file_name are required. If data_tier_name is provided, min_cdate and max_cdate have to be specified and
        the difference in time have to be less than 31 days.

        :param block_name: name of the block
        :type block_name: str
        :param dataset: dataset
        :type dataset: str
        :param data_tier_name: data tier
        :type data_tier_name: str
        :param logical_file_name: Logical File Name
        :type logical_file_name: str
        :param origin_site_name: Origin Site Name (Optional)
        :type origin_site_name: str
        :param run_num: run numbers (Optional). Possible format: run_num, "run_min-run_max", or ["run_min-run_max", run1, run2, ...]
        :type run_num: int, list of runs or list of run ranges
        :param min_cdate: Lower limit for the creation date (unixtime) (Optional)
        :type min_cdate: int, str
        :param max_cdate: Upper limit for the creation date (unixtime) (Optional)
        :type max_cdate: int, str
        :param min_ldate: Lower limit for the last modification date (unixtime) (Optional)
        :type min_ldate: int, str
        :param max_ldate: Upper limit for the last modification date (unixtime) (Optional)
        :type max_ldate: int, str
        :param cdate: creation date (unixtime) (Optional)
        :type cdate: int, str
        :param ldate: last modification date (unixtime) (Optional)
        :type ldate: int, str
        :param detail: Get detailed information of a block (Optional)
        :type detail: bool
        :returns: List of dictionaries containing following keys (block_name). If option detail is used the dictionaries contain the following keys (block_id, create_by, creation_date, open_for_writing, last_modified_by, dataset, block_name, file_count, origin_site_name, last_modification_date, dataset_id and block_size)
        :rtype: list of dicts

        """
        validParameters = ['dataset', 'block_name', 'data_tier_name', 'origin_site_name',
                           'logical_file_name', 'run_num', 'open_for_writing', 'min_cdate',
                           'max_cdate', 'min_ldate', 'max_ldate',
                           'cdate', 'ldate', 'detail']

        #requiredParameters = {'multiple': validParameters}
        requiredParameters = {'multiple': ['dataset', 'block_name', 'data_tier_name', 'logical_file_name']}

        #set defaults
        if 'detail' not in kwargs.keys():
            kwargs['detail'] = False

        checkInputParameter(method="listBlocks", parameters=kwargs.keys(), validParameters=validParameters,
                            requiredParameters=requiredParameters)

        return self.__callServer("blocks", params=kwargs)