def insertProcessingEra(self, businput):
        """
        Input dictionary has to have the following keys:
        processing_version, creation_date,  create_by, description
        it builds the correct dictionary for dao input and executes the dao
        """
        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            businput["processing_era_id"] = self.sm.increment(conn, "SEQ_PE", tran)
            businput["processing_version"] = businput["processing_version"]
            self.pein.execute(conn, businput, tran)
            tran.commit()
            tran = None
        except KeyError as ke:
            dbsExceptionHandler('dbsException-invalid-input',
                                "Invalid input:" + ke.args[0])
        except Exception as ex:
            if (str(ex).lower().find("unique constraint") != -1 or
                str(ex).lower().find("duplicate") != -1):
                        # already exist
                self.logger.warning("DBSProcessingEra/insertProcessingEras. " +
                                "Unique constraint violation being ignored...")
                self.logger.warning(ex)
            else:
                if tran:
                    tran.rollback()
                    tran = None
                raise
        finally:
            if tran:
                tran.rollback()
            if conn:
                conn.close()