def listFileArray(self, **kwargs):
        """
        API to list files in DBS. Non-wildcarded logical_file_name, non-wildcarded dataset, non-wildcarded block_name or non-wildcarded lfn list is required.
        The combination of a non-wildcarded dataset or block_name with an wildcarded logical_file_name is supported.
	

        * For lumi_list the following two json formats are supported:
            - [a1, a2, a3,]
            - [[a,b], [c, d],]
	* lumi_list can be either a list of lumi section numbers as [a1, a2, a3,] or a list of lumi section range as [[a,b], [c, d],]. They cannot be mixed.
        * If lumi_list is provided run only run_num=single-run-number is allowed.
        * When run_num=1, one has to provide logical_file_name. 
        * When lfn list is present, no run or lumi list is allowed.

        :param logical_file_name: logical_file_name of the file, Max length 1000.
        :type logical_file_name: str, list
        :param dataset: dataset
        :type dataset: str
        :param block_name: block name
        :type block_name: str
        :param release_version: release version
        :type release_version: str
        :param pset_hash: parameter set hash
        :type pset_hash: str
        :param app_name: Name of the application
        :type app_name: str
        :param output_module_label: name of the used output module
        :type output_module_label: str
        :param run_num: run , run ranges, and run list, Max list length 1000.
        :type run_num: int, list, string
        :param origin_site_name: site where the file was created
        :type origin_site_name: str
        :param lumi_list: List containing luminosity sections, Max length 1000.
        :type lumi_list: list
        :param detail: Get detailed information about a file
        :type detail: bool
        :param validFileOnly: 0 or 1.  default=0. Return only valid files if set to 1. 
        :type validFileOnly: int
        :param sumOverLumi: 0 or 1.  default=0. When sumOverLumi = 1 and run_num is given , it will count the event by lumi; No list inputs are allowed whtn sumOverLumi=1. 
        :type sumOverLumi: int
        :returns: List of dictionaries containing the following keys (logical_file_name). If detail parameter is true, the dictionaries contain the following keys (check_sum, branch_hash_id, adler32, block_id, event_count, file_type, create_by, logical_file_name, creation_date, last_modified_by, dataset, block_name, file_id, file_size, last_modification_date, dataset_id, file_type_id, auto_cross_section, md5, is_file_valid)
        :rtype: list of dicts

        """
        validParameters = ['dataset', 'block_name', 'logical_file_name',
                          'release_version', 'pset_hash', 'app_name',
                          'output_module_label', 'run_num',
                          'origin_site_name', 'lumi_list', 'detail', 'validFileOnly', 'sumOverLumi']

        requiredParameters = {'multiple': ['dataset', 'block_name', 'logical_file_name']}

        #set defaults
        if 'detail' not in kwargs.keys():
            kwargs['detail'] = False

        checkInputParameter(method="listFileArray", parameters=kwargs.keys(), validParameters=validParameters,
                            requiredParameters=requiredParameters)
        # In order to protect DB and make sure the query can be return in 300 seconds, we limit the length of 
        # logical file names, lumi and run num to 1000. These number may be adjusted later if 
        # needed. YG   May-20-2015.

        # CMS has all MC data with run_num=1. It almost is a full table scan if run_num=1 without lfn. So we will request lfn
        # to be present when run_num=1. YG Jan 14, 2016
        if 'logical_file_name' in kwargs.keys() and isinstance(kwargs['logical_file_name'], list)\
            and len(kwargs['logical_file_name']) > 1:
            if 'run_num' in kwargs.keys() and isinstance(kwargs['run_num'],list) and len(kwargs['run_num']) > 1 :
                raise dbsClientException('Invalid input', 'files API does not supprt two lists: run_num and lfn. ')
            elif 'lumi_list' in kwargs.keys() and kwargs['lumi_list'] and len(kwargs['lumi_list']) > 1 :
                raise dbsClientException('Invalid input', 'files API does not supprt two lists: lumi_lis and lfn. ')
                
        elif 'lumi_list' in kwargs.keys() and kwargs['lumi_list']:
            if 'run_num' not in kwargs.keys() or not kwargs['run_num'] or kwargs['run_num'] ==-1 :
                raise dbsClientException('Invalid input', 'When Lumi section is present, a single run is required. ')
        else:
            if 'run_num' in kwargs.keys():
                if isinstance(kwargs['run_num'], list):
                    if 1 in kwargs['run_num'] or '1' in kwargs['run_num']:
                        raise dbsClientException('Invalid input', 'files API does not supprt run_num=1 when no lumi.')
                else:
                    if kwargs['run_num']==1 or kwargs['run_num']=='1':
                        raise dbsClientException('Invalid input', 'files API does not supprt run_num=1 when no lumi.')

        #check if no lfn is given, but run_num=1 is used for searching
        if ('logical_file_name' not in kwargs.keys() or not kwargs['logical_file_name']) and 'run_num' in kwargs.keys():
            if isinstance(kwargs['run_num'], list):
                if 1 in kwargs['run_num'] or '1' in kwargs['run_num']:
                    raise dbsClientException('Invalid input', 'files API does not supprt run_num=1 without logical_file_name.')
            else:
                if kwargs['run_num'] == 1 or kwargs['run_num'] == '1':
                    raise dbsClientException('Invalid input', 'files API does not supprt run_num=1 without logical_file_name.')
        
        results = []
        mykey = None
        total_lumi_len = 0
        split_lumi_list = []
        max_list_len = 1000 #this number is defined in DBS server
        for key, value in kwargs.iteritems():
            if key == 'lumi_list' and isinstance(kwargs['lumi_list'], list)\
                and kwargs['lumi_list'] and isinstance(kwargs['lumi_list'][0], list):
                lapp = 0
                l = 0
                sm = []
                for i in kwargs['lumi_list']:
                    while i[0]+max_list_len < i[1]:
                        split_lumi_list.append([[i[0], i[0]+max_list_len-1]])
                        i[0] = i[0] + max_list_len
                    else:
                        l += (i[1]-i[0]+1)
                        if l <=  max_list_len:
                            sm.append([i[0], i[1]])
                            lapp = l  #number lumis in sm
                        else:
                            split_lumi_list.append(sm)
                            sm=[]
                            sm.append([i[0], i[1]])
                            lapp = i[1]-i[0]+1
                if sm:
                    split_lumi_list.append(sm)
            elif key in ('logical_file_name', 'run_num', 'lumi_list') and isinstance(value, list) and len(value)>max_list_len:
                mykey =key
#
        if mykey:  
            sourcelist = []
            #create a new list to slice
            sourcelist = kwargs[mykey][:]
            for slice in slicedIterator(sourcelist, max_list_len):
                kwargs[mykey] = slice
                results.extend(self.__callServer("fileArray", data=kwargs, callmethod="POST"))
        elif split_lumi_list:
            for item in split_lumi_list:
                kwargs['lumi_list'] = item
                results.extend(self.__callServer("fileArray", data=kwargs, callmethod="POST"))
        else:
            return self.__callServer("fileArray", data=kwargs, callmethod="POST")
        
        #make sure only one dictionary per lfn.
        #Make sure this changes when we move to 2.7 or 3.0
        #http://stackoverflow.com/questions/11092511/python-list-of-unique-dictionaries
        # YG May-26-2015
        return dict((v['logical_file_name'], v) for v in results).values()