def updateAcqEraEndDate(self, acquisition_era_name ="", end_date=0):
        """
        API to update the end_date of an acquisition era

        :param acquisition_era_name: acquisition_era_name to update (Required)
        :type acquisition_era_name: str
        :param end_date: end_date not zero (Required)
        :type end_date: int

        """
        try:
            self.dbsAcqEra.UpdateAcqEraEndDate( acquisition_era_name, end_date)
        except dbsException as de:
            dbsExceptionHandler(de.eCode, de.message, self.logger.exception, de.message)
        except HTTPError as he:
            raise he
        except Exception as ex:
            sError = "DBSWriterModel/update.AcqEraEndDate %s\n. Exception trace: \n %s" \
                    % (ex, traceback.format_exc())
            dbsExceptionHandler('dbsException-server-error',  dbsExceptionCode['dbsException-server-error'], self.logger.exception, sError)