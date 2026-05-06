def insertPrimaryDataset(self, businput):
        """
        Input dictionary has to have the following keys:
        primary_ds_name, primary_ds_type, creation_date, create_by.
        it builds the correct dictionary for dao input and executes the dao
        """
        conn = self.dbi.connection()
        tran = conn.begin()
        #checking for required fields
        if "primary_ds_name" not in businput:
            dbsExceptionHandler("dbsException-invalid-input",
                " DBSPrimaryDataset/insertPrimaryDataset. " +
                "Primary dataset Name is required for insertPrimaryDataset.")
        try:
            businput["primary_ds_type_id"] = (self.primdstypeList.execute(conn, businput["primary_ds_type"]
                ))[0]["primary_ds_type_id"]
            del businput["primary_ds_type"]
            businput["primary_ds_id"] = self.sm.increment(conn, "SEQ_PDS")
            self.primdsin.execute(conn, businput, tran)
            tran.commit()
            tran = None
        except KeyError as ke:
            dbsExceptionHandler("dbsException-invalid-input",
                " DBSPrimaryDataset/insertPrimaryDataset. Missing: %s" % ke)
            self.logger.warning(" DBSPrimaryDataset/insertPrimaryDataset. Missing: %s" % ke)
        except IndexError as ie:
            dbsExceptionHandler("dbsException-missing-data",
                " DBSPrimaryDataset/insertPrimaryDataset. %s" % ie)
            self.logger.warning(" DBSPrimaryDataset/insertPrimaryDataset. Missing: %s" % ie)
        except Exception as ex:
            if (str(ex).lower().find("unique constraint") != -1 or
                str(ex).lower().find("duplicate") != -1):
                self.logger.warning("DBSPrimaryDataset/insertPrimaryDataset:" +
                        " Unique constraint violation being ignored...")
                self.logger.warning(ex)
            else:
                if tran:
                    tran.rollback()
                if conn: conn.close()
                raise
        finally:
            if tran:
                tran.rollback()
            if conn:
                conn.close()