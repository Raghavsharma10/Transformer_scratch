def listFileArray(self):
        """
        API to list files in DBS. Either non-wildcarded logical_file_name, non-wildcarded dataset, 
	non-wildcarded block_name or non-wildcarded lfn list is required.
        The combination of a non-wildcarded dataset or block_name with an wildcarded logical_file_name is supported.
        
        * For lumi_list the following two json formats are supported:
            - [a1, a2, a3,]
            - [[a,b], [c, d],]
	* lumi_list can be either a list of lumi section numbers as [a1, a2, a3,] or a list of lumi section range as [[a,b], [c, d],]. Thay cannot be mixed.
        * If lumi_list is provided run only run_num=single-run-number is allowed
	* When lfn list is present, no run or lumi list is allowed.
        * When run_num =1 is present, logical_file_name should be present too.

        :param logical_file_name: logical_file_name of the file
        :type logical_file_name: str,  list
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
        :param run_num: run , run ranges, and run list. Possible format are: run_num, 'run_min-run_max' or ['run_min-run_max', run1, run2, ...]. Max length 1000.
        :type run_num: int, list, string
        :param origin_site_name: site where the file was created
        :type origin_site_name: str
        :param lumi_list: List containing luminosity sections. Max length 1000.
        :type lumi_list: list
        :param detail: Get detailed information about a file
        :type detail: bool
        :param validFileOnly: default=0 return all the files. when =1, only return files with is_file_valid=1 or dataset_access_type=PRODUCTION or VALID
        :type validFileOnly: int
        :param sumOverLumi: default=0 event_count is the event_count/file, when=1 and run_num is specified, the event_count is sum of the event_count/lumi for that run; When sumOverLumi = 1, no other input can be a list, for example no run_num list, lumi list or lfn list.
        :type sumOverLumi: int 
        :returns: List of dictionaries containing the following keys (logical_file_name). If detail parameter is true, the dictionaries contain the following keys (check_sum, branch_hash_id, adler32, block_id, event_count, file_type, create_by, logical_file_name, creation_date, last_modified_by, dataset, block_name, file_id, file_size, last_modification_date, dataset_id, file_type_id, auto_cross_section, md5, is_file_valid)
        :rtype: list of dicts

        """
        ret = []
        try :
            body = request.body.read()
            if body:
                data = cjson.decode(body)
                data = validateJSONInputNoCopy("files", data, True)
                if 'sumOverLumi' in data and data['sumOverLumi'] ==1:
                    if ('logical_file_name' in data and isinstance(data['logical_file_name'], list)) \
                       or ('run_num' in data and isinstance(data['run_num'], list)):
                        dbsExceptionHandler("dbsException-invalid-input",
                                            "When sumOverLumi=1, no input can be a list becaue nesting of WITH clause within WITH clause not supported yet by Oracle. ", self.logger.exception)
    
                if 'lumi_list' in data and data['lumi_list']:
                    if 'sumOverLumi' in data and data['sumOverLumi'] ==1:
                        dbsExceptionHandler("dbsException-invalid-input", 
                                            "When lumi_list is given, sumOverLumi must set to 0 becaue nesting of WITH clause within WITH clause not supported yet by Oracle.", self.logger.exception)
                    data['lumi_list'] = self.dbsUtils2.decodeLumiIntervals(data['lumi_list'])	
                    if 'run_num' not in data.keys() or not data['run_num'] or data['run_num'] ==-1 :
                        dbsExceptionHandler("dbsException-invalid-input", 
                                            "When lumi_list is given, require a single run_num.", self.logger.exception)
                #check if run_num =1 w/o lfn 
                if ('logical_file_name' not in data or not data['logical_file_name']) and 'run_num' in data:
                    if isinstance(data['run_num'], list):
                        if 1 in data['run_num'] or '1' in data['run_num']:
                            raise dbsExceptionHandler("dbsException-invalid-input",
                                  'files API does not supprt run_num=1 without logical_file_name.', self.logger.exception)
                        else:
                            if data['run_num'] == 1 or data['run_num'] == '1':
                                raise dbsExceptionHandler("dbsException-invalid-input",
                                   'files API does not supprt run_num=1 without logical_file_name.', self.logger.exception)                
                #Because CMSWEB has a 300 seconds responding time. We have to limit the array siz to make sure that
                #the API can be finished in 300 second. See github issues #465 for tests' results.
                # YG May-20-2015
                max_array_size = 1000
                if ( 'run_num' in data.keys() and isinstance(data['run_num'], list) and len(data['run_num'])>max_array_size)\
                    or ('lumi_list' in data.keys() and isinstance(data['lumi_list'], list) and len(data['lumi_list'])>max_array_size)\
                    or ('logical_file_name' in data.keys() and isinstance(data['logical_file_name'], list) and len(data['logical_file_name'])>max_array_size):
                    dbsExceptionHandler("dbsException-invalid-input", 
                                        "The Max list length supported in listFileArray is %s." %max_array_size, self.logger.exception)
            #   
                ret =  self.dbsFile.listFiles(input_body=data)
        except cjson.DecodeError as De:
            dbsExceptionHandler('dbsException-invalid-input2', "Invalid input", self.logger.exception, str(De))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSReaderModel/listFileArray. %s \n Exception trace: \n %s" \
            % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', ex.message, self.logger.exception, sError)
        for item in ret:
            yield item