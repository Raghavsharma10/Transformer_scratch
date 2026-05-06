def insertAcquisitionEra(self, businput):
        """
        Input dictionary has to have the following keys:
        acquisition_era_name, creation_date, create_by, start_date, end_date.
        it builds the correct dictionary for dao input and executes the dao
        """
        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            businput["acquisition_era_id"] = self.sm.increment(conn, "SEQ_AQE", tran)
            businput["acquisition_era_name"] = businput["acquisition_era_name"]
            #self.logger.warning(businput)
            self.acqin.execute(conn, businput, tran)
            tran.commit()
            tran = None
        except KeyError as ke:
            dbsExceptionHandler('dbsException-invalid-input', "Invalid input:"+ke.args[0])
        except Exception as ex:
            if str(ex).lower().find("unique constraint") != -1 or str(ex).lower().find("duplicate") != -1:
                dbsExceptionHandler('dbsException-invalid-input2', "Invalid input: acquisition_era_name already exists in DB",  serverError="%s" %ex)
            else:
                raise
        finally:
            if tran:
                tran.rollback()
            if conn:
                conn.close()