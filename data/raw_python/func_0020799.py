def insertDatasetWOannex(self, dataset, blockcontent, otptIdList, conn,
                             insertDataset = True, migration = False):
        """
        _insertDatasetOnly_

        Insert the dataset and only the dataset
        Meant to be called after everything else is put into place.

        The insertDataset flag is set to false if the dataset already exists
        """

        tran = conn.begin()
        try:
            #8 Finally, we have everything to insert a dataset
            if insertDataset:
                # Then we have to get a new dataset ID
                dataset['dataset_id'] = self.datasetid.execute(conn,
                                                dataset['dataset'])
                if dataset['dataset_id'] <= 0:
                    dataset['dataset_id'] = self.sm.increment(conn, "SEQ_DS")
                    if not migration:
                        dataset['last_modified_by'] = dbsUtils().getCreateBy()
                        dataset['create_by'] = dbsUtils().getCreateBy()
                        dataset['creation_date'] = dataset.get('creation_date', dbsUtils().getTime())
                        dataset['last_modification_date'] = dataset.get('last_modification_date', dbsUtils().getTime())
                    dataset['xtcrosssection'] = dataset.get('xtcrosssection', None)
                    dataset['prep_id'] = dataset.get('prep_id', None)
                    try:
                        self.datasetin.execute(conn, dataset, tran)
                    except exceptions.IntegrityError as ei:
                        if str(ei).find("ORA-00001") != -1 or str(ei).lower().find("duplicate") !=-1:
                            if conn.closed:
                                conn = self.dbi.connection()
                            dataset['dataset_id'] = self.datasetid.execute(conn, dataset['dataset'])
                            if dataset['dataset_id'] <= 0:
                                if tran:tran.rollback()
                                if conn:conn.close()
                                dbsExceptionHandler('dbsException-conflict-data',
                                                    'Dataset/[processed DS]/[dataset access type] not yet inserted by concurrent insert. ',
                                                    self.logger.exception,
                                                    'Dataset/[processed DS]/[dataset access type] not yet inserted by concurrent insert. '+ str(ei))
                        elif str(ei).find("ORA-01400") > -1:
                            if tran:tran.rollback()
                            if conn:conn.close()
                            dbsExceptionHandler('dbsException-missing-data',
                                'Missing data when insert Datasets. ',
                                self.logger.exception,
                                'Missing data when insert Datasets. '+ str(ei))
                        else:
                            if tran: tran.rollback()
                            if conn: conn.close()
                            dbsExceptionHandler('dbsException-invalid-input2',
                            'Invalid data when insert Datasets. ',
                            self.logger.exception,
                            'Invalid data when insert Datasets. '+ str(ei))

                    except Exception:
                        #should catch all above exception to rollback. YG Jan 17, 2013
                        if tran:tran.rollback()
                        if conn:conn.close()
                        raise

            #9 Fill Dataset Parentage
            #All parentage are deduced from file parentage.

            #10 Before we commit, make dataset and output module configuration
            #mapping.  We have to try to fill the map even if dataset is
            #already in dest db
            for c in otptIdList:
                try:
                    dcObj = {
                             'dataset_id' : dataset['dataset_id'],
                             'output_mod_config_id' : c }
                    self.dcin.execute(conn, dcObj, tran)
                except exceptions.IntegrityError as ei:
                    #FIXME YG 01/17/2013
                    if (str(ei).find("ORA-00001") != -1 and str(ei).find("TUC_DC_1") != -1) or \
                            str(ei).lower().find("duplicate")!=-1:
                    #ok, already in db
                    #FIXME: What happens when there are partially in db?
                    #YG 11/17/2010
                        pass
                    else:
                        if tran:tran.rollback()
                        if conn:conn.close()
                        dbsExceptionHandler('dbsException-invalid-input2',
                            'Invalid data when insert dataset_configs. ',
                            self.logger.exception,
                            'Invalid data when insert dataset_configs. '+ str(ei))
                except Exception as ex:
                    if tran:tran.rollback()
                    if conn:conn.close()
                    raise
            #Now commit everything.
            tran.commit()
        except exceptions.IntegrityError as ei:
            # Then is it already in the database?
            # Not really. We have to check it again. YG Jan 17, 2013
            # we don't check the unique key here, since there are more than one unique key might
            # be violated: such as data_tier, processed_dataset, dataset_access_types.
            if str(ei).find("ORA-00001") != -1 or str(ei).lower().find("duplicate")!=-1:
                # For now, we assume most cases are the same dataset was instered by different thread. If not,
                # one has to call the insert dataset again. But we think this is a rare case and let the second
                # DBSBlockInsert call fix it if it happens.
                if conn.closed:
                    conn = self.dbi.connection()
                dataset_id = self.datasetid.execute(conn, dataset['dataset'])
                if dataset_id <= 0:
                    dbsExceptionHandler('dbsException-conflict-data',
                                        'Dataset not yet inserted by concurrent insert',
                                        self.logger.exception,
                                        'Dataset not yet inserted by concurrent insert')

                else:
                    dataset['dataset_id'] = dataset_id
            else:
                if tran:tran.rollback()
                if conn:conn.close()
                dbsExceptionHandler('dbsException-invalid-input2',
                    'Invalid data when insert Datasets. ',
                    self.logger.exception,
                    'Invalid data when insert Datasets. '+ str(ei))
        except Exception as ex:
            if tran:tran.rollback()
            if conn:conn.close()
            raise
        finally:
            if tran:tran.rollback()
            if conn:conn.close()
        return dataset['dataset_id']