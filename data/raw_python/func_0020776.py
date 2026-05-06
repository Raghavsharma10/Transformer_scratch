def listFileParentsByLumi(self):
        """
        IMPORTANT: This is ***WMAgent*** sepcial case API. It is not for others.

        API to list File Parentage  for a given block with or w/o a list of LFN. It is used with the POST method of fileparents call.
        Using the child_lfn_list will significantly affect the API running speed. 
        
        :param block_name: This this the child's block name 
        :type  block_name: str
        :param logical_file_name: is a list of child lfn.  Max length 1000.  
        :type  logial_file_name: list of str
        :returns: List of dictionaries containing the following keys:[{child_parent_id_list: [(cid1, pid1), (cid2, pid2), ... (cidn, pidn)]}]
        :rtype: list of dicts
        
        """
        try :
            body = request.body.read()
            if body:
                data = cjson.decode(body)
                data = validateJSONInputNoCopy('file_parent_lumi', data, read=True)
            else:
                data = {}

            #Because CMSWEB has a 300 seconds responding time. We have to limit the array siz to make sure that
            #the API can be finished in 300 second. 
            max_array_size = 1000
            if ('logical_file_name' in data.keys() and isinstance(data['logical_file_name'], list) and len(data['logical_file_name'])>max_array_size):
                dbsExceptionHandler("dbsException-invalid-input",
                                        "The Max list length supported in listFilePArentsByLumi is %s." %max_array_size, self.logger.exception)

            lfn = []
            if "block_name" not in data.keys():
                 dbsExceptionHandler('dbsException-invalid-input', "block_name is required for fileparentsbylumi")
            else:
                if "logical_file_name" in data.keys():
                    lfn = data["logical_file_name"]
            result = self.dbsFile.listFileParentsByLumi(block_name=data["block_name"], logical_file_name=lfn)
            for r in result:
                yield r
        except cjson.DecodeError as De:
            dbsExceptionHandler('dbsException-invalid-input2', "Invalid input", self.logger.exception, str(De))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSReaderModel/listFileParentsByLumi. %s \n Exception trace: \n %s" \
            % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', ex.message, self.logger.exception, sError)