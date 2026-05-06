def insertPrimaryDataset(self):
        """
        API to insert A primary dataset in DBS

        :param primaryDSObj: primary dataset object
        :type primaryDSObj: dict
        :key primary_ds_type: TYPE (out of valid types in DBS, MC, DATA) (Required)
        :key primary_ds_name: Name of the primary dataset (Required)

        """
        try :
            body = request.body.read()
            indata = cjson.decode(body)
            indata = validateJSONInputNoCopy("primds", indata)
            indata.update({"creation_date": dbsUtils().getTime(), "create_by": dbsUtils().getCreateBy() })
            self.dbsPrimaryDataset.insertPrimaryDataset(indata)
        except cjson.DecodeError as dc:
            dbsExceptionHandler("dbsException-invalid-input2", "Wrong format/data from insert PrimaryDataset input",  self.logger.exception, str(dc))
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.message)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSWriterModel/insertPrimaryDataset. %s\n Exception trace: \n %s" \
                        % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)