def listDatasets(self, **kwargs):
        """
        API to list dataset(s) in DBS
        * You can use ANY combination of these parameters in this API
        * In absence of parameters, all valid datasets known to the DBS instance will be returned

        :param dataset:  Full dataset (path) of the dataset
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
        :param processing_version: Processing Version
        :type processing_version: str
        :param acquisition_era_name: Acquisition Era
        :type acquisition_era_name: str
        :param run_num: Specify a specific run number or range: Possible format: run_num, "run_min-run_max", or ["run_min-run_max", run1, run2, ...]
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
	:param dataset_id: DB primary key of datasets table.
        :type dataset_id: int, str	
        :returns: List of dictionaries containing the following keys (dataset). If the detail option is used. The dictionary contain the following keys (primary_ds_name, physics_group_name, acquisition_era_name, create_by, dataset_access_type, data_tier_name, last_modified_by, creation_date, processing_version, processed_ds_name, xtcrosssection, last_modification_date, dataset_id, dataset, prep_id, primary_ds_type)
        :rtype: list of dicts

        """
        validParameters = ['dataset', 'parent_dataset', 'is_dataset_valid',
                           'release_version', 'pset_hash', 'app_name',
                           'output_module_label', 'processing_version', 'acquisition_era_name',
                           'run_num', 'physics_group_name', 'logical_file_name',
                           'primary_ds_name', 'primary_ds_type', 'processed_ds_name', 'data_tier_name',
                           'dataset_access_type', 'prep_id', 'create_by', 'last_modified_by',
                           'min_cdate', 'max_cdate', 'min_ldate', 'max_ldate', 'cdate', 'ldate',
                           'detail', 'dataset_id']

        #set defaults
        if 'detail' not in kwargs.keys():
            kwargs['detail'] = False

        checkInputParameter(method="listDatasets", parameters=kwargs.keys(), validParameters=validParameters)

        return self.__callServer("datasets", params=kwargs)