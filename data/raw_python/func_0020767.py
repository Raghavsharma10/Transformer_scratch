def listDatasetArray(self):
        """
        API to list datasets in DBS. To be called by datasetlist url with post call.

        :param dataset: list of datasets [dataset1,dataset2,..,dataset n] (must have either a list of dataset or dataset_id), Max length 1000.
        :type dataset: list
	:param dataset_id: list of dataset ids [dataset_id1,dataset_id2,..,dataset_idn, "dsid_min-dsid_max"] ((must have either a list of dataset or dataset_id)
        :type dataset_id: list
        :param dataset_access_type: List only datasets with that dataset access type (Optional)
        :type dataset_access_type: str
        :param detail: brief list or detailed list 1/0
        :type detail: bool
        :returns: List of dictionaries containing the following keys (dataset). If the detail option is used. The dictionary contains the following keys (primary_ds_name, physics_group_name, acquisition_era_name, create_by, dataset_access_type, data_tier_name, last_modified_by, creation_date, processing_version, processed_ds_name, xtcrosssection, last_modification_date, dataset_id, dataset, prep_id, primary_ds_type)
        :rtype: list of dicts

        """
        ret = []
        try :
            body = request.body.read()
            if body:
                data = cjson.decode(body)
                data = validateJSONInputNoCopy("dataset", data, read=True)
                #Because CMSWEB has a 300 seconds responding time. We have to limit the array siz to make sure that
                #the API can be finished in 300 second. 
                # YG Nov-05-2015
                max_array_size = 1000
                if ( 'dataset' in data.keys() and isinstance(data['dataset'], list) and len(data['dataset'])>max_array_size)\
                    or ('dataset_id' in data.keys() and isinstance(data['dataset_id'], list) and len(data['dataset_id'])>max_array_size):
                    dbsExceptionHandler("dbsException-invalid-input",
                                        "The Max list length supported in listDatasetArray is %s." %max_array_size, self.logger.exception)    
                ret = self.dbsDataset.listDatasetArray(data)
        except cjson.DecodeError as De:
            dbsExceptionHandler('dbsException-invalid-input2', "Invalid input", self.logger.exception, str(De))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSReaderModel/listDatasetArray. %s \n Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)
        for item in ret:
                    yield item