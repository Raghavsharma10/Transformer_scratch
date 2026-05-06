def listDataTiers(self, data_tier_name=""):
        """
        API to list data tiers known to DBS.

        :param data_tier_name: List details on that data tier (Optional)
        :type data_tier_name: str
        :returns: List of dictionaries containing the following keys (data_tier_id, data_tier_name, create_by, creation_date)

        """
        data_tier_name = data_tier_name.replace("*", "%")

        try:
            conn = self.dbi.connection()
            return self.dbsDataTierListDAO.execute(conn, data_tier_name.upper())
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.message)
        except ValueError as ve:
            dbsExceptionHandler("dbsException-invalid-input2", "Invalid Input Data",  self.logger.exception, ve.message)
        except TypeError as te:
            dbsExceptionHandler("dbsException-invalid-input2", "Invalid Input DataType",  self.logger.exception, te.message)
        except NameError as ne:
            dbsExceptionHandler("dbsException-invalid-input2", "Invalid Input Searching Key",  self.logger.exception, ne.message)
        except Exception as ex:
            sError = "DBSReaderModel/listDataTiers. %s\n. Exception trace: \n %s" \
                    % ( ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)
        finally:
            if conn:
                conn.close()