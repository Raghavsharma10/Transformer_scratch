def insertOutputModuleConfig(self, remoteConfig, migration=False):
        """
        Insert Release version, application, parameter set hashes and the map(output module config).

        """
        otptIdList = []
        missingList = []
        conn = self.dbi.connection()
        try:
            for c in remoteConfig:
                cfgid = self.otptModCfgid.execute(conn, app = c["app_name"],
                                      release_version = c["release_version"],
                                      pset_hash = c["pset_hash"],
                                      output_label = c["output_module_label"],
                                      global_tag=c['global_tag'])
                if cfgid <= 0 :
                    missingList.append(c)
                else:
                    key = (c['app_name'] + ':' + c['release_version'] + ':' +
                           c['pset_hash'] + ':' +
                           c['output_module_label'] + ':' + c['global_tag'])
                    self.datasetCache['conf'][key] = cfgid
                    otptIdList.append(cfgid)
                    #print "About to set cfgid: %s" % str(cfgid)
        except KeyError as ex:
            if conn:conn.close()
            dbsExceptionHandler("dbsException-invalid-input2", "DBSBlockInsert/insertOutputModuleConfig: \
                KeyError exception: %s. " %ex.args[0], self.logger.exception,
	        "DBSBlockInsert/insertOutputModuleConfig: KeyError exception: %s. " %ex.args[0]	)
        except Exception as ex:
            if conn:conn.close()
            raise

        if len(missingList)==0:
            if conn:conn.close()
            return otptIdList
        #Now insert the missing configs
        try:
            #tran = conn.begin()
            for m in missingList:
                # Start a new transaction
                # This is to see if we can get better results
                # by committing early if we're submitting
                # multiple blocks with similar features
                tran = conn.begin()
                #Now insert the config
                # Sort out the mess
                # We're having some problems with different threads
                # committing different pieces at the same time
                # This makes the output module config ID wrong
                # Trying to catch this via exception handling on duplication
                # Start a new transaction
                #global_tag is now required. YG 03/08/2011
                try:
                    cfgid = 0
                    if not migration:
                        m['create_by'] = dbsUtils().getCreateBy()
                        m['creation_date'] = dbsUtils().getTime()
                    configObj = {"release_version": m["release_version"],
                                 "pset_hash": m["pset_hash"], "pset_name":m.get('pset_name', None),
                                 "app_name": m["app_name"],
                                 'output_module_label' : m['output_module_label'],
                                 'global_tag' : m['global_tag'],
                                 'scenario' : m.get('scenario', None),
                                 'creation_date' : m['creation_date'],
                                 'create_by':m['create_by']
                                  }
                    self.otptModCfgin.execute(conn, configObj, tran)
                    tran.commit()
                    tran = None
                except KeyError as ex:
                    if tran:tran.rollback()
                    if conn:conn.close()
                    dbsExceptionHandler("dbsException-invalid-input2", "DBSBlockInsert/insertOutputModuleConfig: \
                                         KeyError exception: %s. " %ex.args[0],
					 self.logger.exception, 
					"DBSBlockInsert/insertOutputModuleConfig: KeyError exception: %s. " %ex.args[0])
                except exceptions.IntegrityError as ex:
                    #Another job inserted it just 1/100000 second earlier than
                    #you!!  YG 11/17/2010
                    if str(ex).find("ORA-00001") != -1 or str(ex).lower().find("duplicate") !=-1:
                        if str(ex).find("TUC_OMC_1") != -1:
                            #the config is already in db, get the ID later
                            pass
                        else:
                            #reinsert it if one or two or three of the three attributes (vresion, hash and app) are inserted
                            #just 1/100000 second eailer.
                            try:
                                self.otptModCfgin.execute(conn, configObj, tran)
                                tran.commit()
                                tran = None
                            except exceptions.IntegrityError as ex:
                                if (str(ex).find("ORA-00001") != -1 and str(ex).find("TUC_OMC_1"))\
                                        or str(ex).lower().find("duplicate") != -1:
                                    pass
                                else:
                                    if tran:tran.rollback()
                                    if conn:conn.close()
                                    dbsExceptionHandler('dbsException-invalid-input2',
                                        'Invalid data when insert Configure. ',
                                        self.logger.exception,
                                        'Invalid data when insert Configure. '+ str(ex))
                    elif str(ex).find("ORA-01400") > -1:
                        if tran:tran.rollback()
                        if conn:conn.close()
                        dbsExceptionHandler("dbsException-missing-data", "Missing data when inserting Configure. ", 
				self.logger.exception, str(ex))
                    else:
                        if tran:tran.rollback()
                        if conn:conn.close()
                        dbsExceptionHandler('dbsException-invalid-input2',
                            'Invalid data when insert Configure. ',
                            self.logger.exception,
                            'Invalid data when insert Configure. '+ str(ex))
                except exceptions as ex3:
                    if tran:tran.rollback()
                    if conn:conn.close()
                    raise ex3
                cfgid = self.otptModCfgid.execute(conn,
                                    app = m["app_name"],
                                    release_version = m["release_version"],
                                    pset_hash = m["pset_hash"],
                                    output_label = m["output_module_label"],
                                    global_tag=m['global_tag'])
                otptIdList.append(cfgid)
                key = (m['app_name'] + ':' + m['release_version'] + ':' +
                       m['pset_hash'] + ':' +m['output_module_label'] + ':' +
                       m['global_tag'])
                self.datasetCache['conf'][key] = cfgid
        finally:
            if tran:tran.rollback()
            if conn:conn.close()
        return otptIdList