def listDatasets(self, dataset="", parent_dataset="", is_dataset_valid=1,
        release_version="", pset_hash="", app_name="", output_module_label="", global_tag="",
        processing_version=0, acquisition_era_name="", run_num=-1,
        physics_group_name="", logical_file_name="", primary_ds_name="", primary_ds_type="",
        processed_ds_name='', data_tier_name="", dataset_access_type="VALID", prep_id='', create_by="", last_modified_by="",
        min_cdate='0', max_cdate='0', min_ldate='0', max_ldate='0', cdate='0',
        ldate='0', detail=False, dataset_id=-1):
        """
        API to list dataset(s) in DBS
        * You can use ANY combination of these parameters in this API
        * In absence of parameters, all valid datasets known to the DBS instance will be returned

        :param dataset:  Full dataset (path) of the dataset.
        :type dataset: str
        :param parent_dataset: Full dataset (path) of the dataset
        :type parent_dataset: str
        :param release_version: cmssw version
        :type release_version: str
        :param pset_hash: pset hash
        :type pset_hash: str
        :param app_name: Application name (generally it is cmsRun)
        :type app_name: str
        :param output_module_label: output_module_label
        :type output_module_label: str
        :param global_tag: global_tag
        :type global_tag: str
        :param processing_version: Processing Version
        :type processing_version: str
        :param acquisition_era_name: Acquisition Era
        :type acquisition_era_name: str
        :param run_num: Specify a specific run number or range. Possible format are: run_num, 'run_min-run_max' or ['run_min-run_max', run1, run2, ...]. run_num=1 is not allowed.
        :type run_num: int,list,str
        :param physics_group_name: List only dataset having physics_group_name attribute
        :type physics_group_name: str
        :param logical_file_name: List dataset containing the logical_file_name
        :type logical_file_name: str
        :param primary_ds_name: Primary Dataset Name
        :type primary_ds_name: str
        :param primary_ds_type: Primary Dataset Type (Type of data, MC/DATA)
        :type primary_ds_type: str
        :param processed_ds_name: List datasets having this processed dataset name
        :type processed_ds_name: str
        :param data_tier_name: Data Tier
        :type data_tier_name: str
        :param dataset_access_type: Dataset Access Type ( PRODUCTION, DEPRECATED etc.)
        :type dataset_access_type: str
        :param prep_id: prep_id
        :type prep_id: str
        :param create_by: Creator of the dataset
        :type create_by: str
        :param last_modified_by: Last modifier of the dataset
        :type last_modified_by: str
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
        :param detail: List all details of a dataset
        :type detail: bool
        :param dataset_id: dataset table primary key used by CMS Computing Analytics.
        :type dataset_id: int, long, str
        :returns: List of dictionaries containing the following keys (dataset). If the detail option is used. The dictionary contain the following keys (primary_ds_name, physics_group_name, acquisition_era_name, create_by, dataset_access_type, data_tier_name, last_modified_by, creation_date, processing_version, processed_ds_name, xtcrosssection, last_modification_date, dataset_id, dataset, prep_id, primary_ds_type)
        :rtype: list of dicts

        """
        dataset = dataset.replace("*", "%")
        parent_dataset = parent_dataset.replace("*", "%")
        release_version = release_version.replace("*", "%")
        pset_hash = pset_hash.replace("*", "%")
        app_name = app_name.replace("*", "%")
        output_module_label = output_module_label.replace("*", "%")
        global_tag = global_tag.replace("*", "%")
        logical_file_name = logical_file_name.replace("*", "%")
        physics_group_name = physics_group_name.replace("*", "%")
        primary_ds_name = primary_ds_name.replace("*", "%")
        primary_ds_type = primary_ds_type.replace("*", "%")
        data_tier_name = data_tier_name.replace("*", "%")
        dataset_access_type = dataset_access_type.replace("*", "%")
        processed_ds_name = processed_ds_name.replace("*", "%")
        acquisition_era_name = acquisition_era_name.replace("*", "%")
        #processing_version =  processing_version.replace("*", "%")
        #create_by and last_modified_by have be full spelled, no wildcard will allowed.
        #We got them from request head so they can be either HN account name or DN.
        #This is depended on how an user's account is set up.
        #
        # In the next release we will require dataset has no wildcard in it. 
        # DBS will reject wildcard search with dataset name with listDatasets call. 
        # One should seperate the dataset into primary , process and datatier if any wildcard.
        # YG Oct 26, 2016
        # Some of users were overwhiled by the API change. So we split the wildcarded dataset in the server instead of by the client.
        # YG Dec. 9 2016
        #
        # run_num=1 caused full table scan and CERN DBS reported some of the queries ran more than 50 hours
        # We will disbale all the run_num=1 calls in DBS. Run_num=1 will be OK when logical_file_name is given.
        # YG Jan. 15 2019
        #
        if (run_num != -1 and logical_file_name ==''):
            for r in parseRunRange(run_num):
                if isinstance(r, basestring) or isinstance(r, int) or isinstance(r, long):    
                    if r == 1 or r == '1':
                        dbsExceptionHandler("dbsException-invalid-input", "Run_num=1 is not a valid input.",
                                self.logger.exception)
                elif isinstance(r, run_tuple):
                    if r[0] == r[1]:
                        dbsExceptionHandler('dbsException-invalid-input', "DBS run range must be apart at least by 1.", 
                          self.logger.exception)
                    elif r[0] <= 1 <= r[1]:
                        dbsExceptionHandler("dbsException-invalid-input", "Run_num=1 is not a valid input.",
                                self.logger.exception)     

        if( dataset and ( dataset == "/%/%/%" or dataset== "/%" or dataset == "/%/%" ) ):
            dataset=''
        elif( dataset and ( dataset.find('%') != -1 ) ) :
            junk, primary_ds_name, processed_ds_name, data_tier_name = dataset.split('/')
            dataset = ''
        if ( primary_ds_name == '%' ):
            primary_ds_name = ''
        if( processed_ds_name == '%' ):
            processed_ds_name = ''
        if ( data_tier_name == '%' ):
            data_tier_name = ''

        try:
            dataset_id = int(dataset_id)
        except:
            dbsExceptionHandler("dbsException-invalid-input2", "Invalid Input for dataset_id that has to be an int.",
                                self.logger.exception, 'dataset_id has to be an int.')
        if create_by.find('*')!=-1 or create_by.find('%')!=-1 or last_modified_by.find('*')!=-1\
                or last_modified_by.find('%')!=-1:
            dbsExceptionHandler("dbsException-invalid-input2", "Invalid Input for create_by or last_modified_by.\
            No wildcard allowed.",  self.logger.exception, 'No wildcards allowed for create_by or last_modified_by')
        try:
            if isinstance(min_cdate, basestring) and ('*' in min_cdate or '%' in min_cdate):
                min_cdate = 0
            else:
                try:
                    min_cdate = int(min_cdate)
                except:
                    dbsExceptionHandler("dbsException-invalid-input", "invalid input for min_cdate")
            
            if isinstance(max_cdate, basestring) and ('*' in max_cdate or '%' in max_cdate):
                max_cdate = 0
            else:
                try:
                    max_cdate = int(max_cdate)
                except:
                    dbsExceptionHandler("dbsException-invalid-input", "invalid input for max_cdate")
            
            if isinstance(min_ldate, basestring) and ('*' in min_ldate or '%' in min_ldate):
                min_ldate = 0
            else:
                try:
                    min_ldate = int(min_ldate)
                except:
                    dbsExceptionHandler("dbsException-invalid-input", "invalid input for min_ldate")
            
            if isinstance(max_ldate, basestring) and ('*' in max_ldate or '%' in max_ldate):
                max_ldate = 0
            else:
                try:
                    max_ldate = int(max_ldate)
                except:
                    dbsExceptionHandler("dbsException-invalid-input", "invalid input for max_ldate")
            
            if isinstance(cdate, basestring) and ('*' in cdate or '%' in cdate):
                cdate = 0
            else:
                try:
                    cdate = int(cdate)
                except:
                    dbsExceptionHandler("dbsException-invalid-input", "invalid input for cdate")
            
            if isinstance(ldate, basestring) and ('*' in ldate or '%' in ldate):
                ldate = 0
            else:
                try:
                    ldate = int(ldate)
                except:
                    dbsExceptionHandler("dbsException-invalid-input", "invalid input for ldate")
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listDatasets.  %s \n. Exception trace: \n %s" \
                % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)

        detail = detail in (True, 1, "True", "1", 'true')
        try: 
            return self.dbsDataset.listDatasets(dataset, parent_dataset, is_dataset_valid, release_version, pset_hash,
                app_name, output_module_label, global_tag, processing_version, acquisition_era_name, 
                run_num, physics_group_name, logical_file_name, primary_ds_name, primary_ds_type, processed_ds_name,
                data_tier_name, dataset_access_type, prep_id, create_by, last_modified_by,
                min_cdate, max_cdate, min_ldate, max_ldate, cdate, ldate, detail, dataset_id)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except Exception as ex:
            sError = "DBSReaderModel/listdatasets. %s.\n Exception trace: \n %s" % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)