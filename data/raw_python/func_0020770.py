def listBlocksParents(self):
        """
        API to list block parents of multiple blocks. To be called by blockparents url with post call.

        :param block_names: list of block names [block_name1, block_name2, ...] (Required). Mwx length 1000.
        :type block_names: list

        """
        try :
            body = request.body.read()
            data = cjson.decode(body)
            data = validateJSONInputNoCopy("block", data, read=True)
            #Because CMSWEB has a 300 seconds responding time. We have to limit the array siz to make sure that
            #the API can be finished in 300 second. 
            # YG Nov-05-2015
            max_array_size = 1000
            if ( 'block_names' in data.keys() and isinstance(data['block_names'], list) and len(data['block_names'])>max_array_size):
                    dbsExceptionHandler("dbsException-invalid-input",
                                        "The Max list length supported in listBlocksParents is %s." %max_array_size, self.logger.exception)
            return self.dbsBlock.listBlockParents(data["block_name"])
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.serverError)
        except cjson.DecodeError as de:
            sError = "DBSReaderModel/listBlockParents. %s\n. Exception trace: \n %s" \
                    % (de, traceback.format_exc())
            msg = "DBSReaderModel/listBlockParents. %s" % de
            dbsExceptionHandler('dbsException-invalid-input2', msg, self.logger.exception, sError)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSReaderModel/listBlockParents. %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)