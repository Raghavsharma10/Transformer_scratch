def getServices(self):
        """
        Simple method that returs list of all know DBS instances, instances known to this registry
        """
        try:
            conn = self.dbi.connection()
            result = self.serviceslist.execute(conn)
            return result
        except Exception as ex:
            msg = (("%s DBSServicesRegistry/getServices." + 
                    " %s\n. Exception trace: \n %s") %
                   (DBSEXCEPTIONS['dbsException-3'], ex,
                    traceback.format_exc()))
            self.logger.exception(msg )
            raise Exception ("dbsException-3", msg )
        finally:
            conn.close()