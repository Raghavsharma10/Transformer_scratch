def remove(self):
        """
        Interface to remove a migration request from the queue.
        Only Permanent FAILED/9 and PENDING/0 requests can be removed
        (running and sucessed requests cannot be removed)

        """
        body = request.body.read()
        indata = cjson.decode(body)
        try:
            indata = validateJSONInputNoCopy("migration_rqst", indata)
            return self.dbsMigrate.removeMigrationRequest(indata)
        except dbsException as he:
            dbsExceptionHandler(he.eCode, he.message, self.logger.exception, he.message)
        except Exception as e:
            if e.code == 400:
                dbsExceptionHandler('dbsException-invalid-input2', str(e), self.logger.exception, str(e))    
            else:
                dbsExceptionHandler('dbsException-server-error', dbsExceptionCode['dbsException-server-error'], self.logger.exception, str(e))