def insertDataset(self, businput):
        """
        input dictionary must have the following keys:
        dataset, primary_ds_name(name), processed_ds(name), data_tier(name),
        acquisition_era(name), processing_version
        It may have following keys:
        physics_group(name), xtcrosssection, creation_date, create_by, 
        last_modification_date, last_modified_by
        """ 
        if not ("primary_ds_name" in businput and "dataset" in businput
                and "dataset_access_type" in businput and "processed_ds_name" in businput ):
            dbsExceptionHandler('dbsException-invalid-input', "business/DBSDataset/insertDataset must have dataset,\
                dataset_access_type, primary_ds_name, processed_ds_name as input")

        if "data_tier_name" not in businput:
            dbsExceptionHandler('dbsException-invalid-input', "insertDataset must have data_tier_name as input.")

        conn = self.dbi.connection()
        tran = conn.begin()
        try:

            dsdaoinput = {}
            dsdaoinput["primary_ds_name"] = businput["primary_ds_name"]
            dsdaoinput["data_tier_name"] =  businput["data_tier_name"].upper()
            dsdaoinput["dataset_access_type"] = businput["dataset_access_type"].upper()
            #not required pre-exist in the db. will insert with the dataset if not in yet
            #processed_ds_name=acquisition_era_name[-fileter_name][-processing_str]-vprocessing_version   Changed as 4/30/2012 YG.
            #althrough acquisition era and processing version is not required for a dataset in the schema(the schema is build this way because
            #we need to accomdate the DBS2 data), but we impose the requirement on the API. So both acquisition and processing eras are required 
            #YG 12/07/2011  TK-362
            if "acquisition_era_name" in businput and "processing_version" in businput:
                erals=businput["processed_ds_name"].rsplit('-')
                if erals[0]==businput["acquisition_era_name"] and erals[len(erals)-1]=="%s%s"%("v", businput["processing_version"]):
                    dsdaoinput["processed_ds_name"] = businput["processed_ds_name"]
                else:
                    dbsExceptionHandler('dbsException-invalid-input', "insertDataset:\
                    processed_ds_name=acquisition_era_name[-filter_name][-processing_str]-vprocessing_version must be satisified.")
            else:
                dbsExceptionHandler("dbsException-missing-data",  "insertDataset: Required acquisition_era_name or processing_version is not found in the input")
            
            if "physics_group_name" in businput:
                dsdaoinput["physics_group_id"] = self.phygrpid.execute(conn, businput["physics_group_name"])
                if dsdaoinput["physics_group_id"]  == -1:
                    dbsExceptionHandler("dbsException-missing-data",  "insertDataset. physics_group_name not found in DB")
            else:
                dsdaoinput["physics_group_id"] = None

            dsdaoinput["dataset_id"] = self.sm.increment(conn, "SEQ_DS")
            # we are better off separating out what we need for the dataset DAO
            dsdaoinput.update({ 
                               "dataset" : "/%s/%s/%s" %
                               (businput["primary_ds_name"],
                                businput["processed_ds_name"],
                                businput["data_tier_name"].upper()),
                               "prep_id" : businput.get("prep_id", None),
                               "xtcrosssection" : businput.get("xtcrosssection", None),
                               "creation_date" : businput.get("creation_date", dbsUtils().getTime() ),
                               "create_by" : businput.get("create_by", dbsUtils().getCreateBy()) ,
                               "last_modification_date" : businput.get("last_modification_date", dbsUtils().getTime()),
                               #"last_modified_by" : businput.get("last_modified_by", dbsUtils().getModifiedBy())
                               "last_modified_by" : dbsUtils().getModifiedBy()
                               })
            """
            repeated again, why?  comment out by YG 3/14/2012
            #physics group
            if "physics_group_name" in businput:
                dsdaoinput["physics_group_id"] = self.phygrpid.execute(conn, businput["physics_group_name"])
                if dsdaoinput["physics_group_id"]  == -1:
                    dbsExceptionHandler("dbsException-missing-data",  "insertDataset. Physics Group : %s Not found"
                                                                                    % businput["physics_group_name"])
            else: dsdaoinput["physics_group_id"] = None
            """
            # See if Processing Era exists
            if "processing_version" in businput and businput["processing_version"] != 0:
                dsdaoinput["processing_era_id"] = self.proceraid.execute(conn, businput["processing_version"])
                if dsdaoinput["processing_era_id"] == -1 :
                    dbsExceptionHandler("dbsException-missing-data", "DBSDataset/insertDataset: processing_version not found in DB") 
            else:
                dbsExceptionHandler("dbsException-invalid-input", "DBSDataset/insertDataset: processing_version is required")

            # See if Acquisition Era exists
            if "acquisition_era_name" in businput:
                dsdaoinput["acquisition_era_id"] = self.acqeraid.execute(conn, businput["acquisition_era_name"])
                if dsdaoinput["acquisition_era_id"] == -1:
                    dbsExceptionHandler("dbsException-missing-data", "DBSDataset/insertDataset: acquisition_era_name not found in DB")
            else:
                dbsExceptionHandler("dbsException-invalid-input", "DBSDataset/insertDataset:  acquisition_era_name is required")
            try:
                # insert the dataset
                self.datasetin.execute(conn, dsdaoinput, tran)
            except SQLAlchemyIntegrityError as ex:
                if (str(ex).lower().find("unique constraint") != -1 or
                    str(ex).lower().find("duplicate") != -1):
                    # dataset already exists, lets fetch the ID
                    self.logger.warning(
                            "Unique constraint violation being ignored...")
                    self.logger.warning("%s" % ex)
                    ds = "/%s/%s/%s" % (businput["primary_ds_name"], businput["processed_ds_name"], businput["data_tier_name"].upper())
                    dsdaoinput["dataset_id"] = self.datasetid.execute(conn, ds )
                    if dsdaoinput["dataset_id"] == -1 :
                        dbsExceptionHandler("dbsException-missing-data", "DBSDataset/insertDataset. Strange error, the dataset %s does not exist ?" 
                                                    % ds )
                if (str(ex).find("ORA-01400") ) != -1 :
                    dbsExceptionHandler("dbsException-missing-data", "insertDataset must have: dataset,\
                                          primary_ds_name, processed_ds_name, data_tier_name ")
            except Exception as e:
                raise       

            #FIXME : What about the READ-only status of the dataset
            #There is no READ-oly status for a dataset.

            # Create dataset_output_mod_mod_configs mapping
            if "output_configs" in businput:
                for anOutConfig in businput["output_configs"]:
                    dsoutconfdaoin = {}
                    dsoutconfdaoin["dataset_id"] = dsdaoinput["dataset_id"]
                    dsoutconfdaoin["output_mod_config_id"] = self.outconfigid.execute(conn, anOutConfig["app_name"],
                                                                                anOutConfig["release_version"],
                                                                                anOutConfig["pset_hash"],
                                                                                anOutConfig["output_module_label"],
                                                                                anOutConfig["global_tag"]) 
                    if dsoutconfdaoin["output_mod_config_id"] == -1 : 

                        dbsExceptionHandler("dbsException-missing-data", "DBSDataset/insertDataset: Output config (%s, %s, %s, %s, %s) not found"
                                                                                % (anOutConfig["app_name"],
                                                                                   anOutConfig["release_version"],
                                                                                   anOutConfig["pset_hash"],
                                                                                   anOutConfig["output_module_label"],
                                                                                   anOutConfig["global_tag"]))
                    try:
                        self.datasetoutmodconfigin.execute(conn, dsoutconfdaoin, tran)
                    except Exception as ex:
                        if str(ex).lower().find("unique constraint") != -1 or str(ex).lower().find("duplicate") != -1:
                            pass
                        else:
                            raise
            # Dataset parentage will NOT be added by this API it will be set by insertFiles()--deduced by insertFiles
            # Dataset  runs will NOT be added by this API they will be set by insertFiles()--deduced by insertFiles OR insertRun API call
            tran.commit()
            tran = None
        except Exception:
            if tran:
                tran.rollback()
                tran = None
            raise
        finally:
            if tran:
                tran.rollback()
            if conn:
                conn.close()