def updateStatus(self, block_name="", open_for_writing=0):
        """
        Used to toggle the status of a block open_for_writing=1, open for writing, open_for_writing=0, closed
        """
        if open_for_writing not in [1, 0, '1', '0']:
            msg = "DBSBlock/updateStatus. open_for_writing can only be 0 or 1 : passed %s."\
                   % open_for_writing 
            dbsExceptionHandler('dbsException-invalid-input', msg)
        conn = self.dbi.connection()
        trans = conn.begin()
        try :
            open_for_writing = int(open_for_writing)
            self.updatestatus.execute(conn, block_name, open_for_writing, dbsUtils().getTime(), trans)
            trans.commit()
            trans = None
        except Exception as ex:
            if trans:
                trans.rollback()
            if conn:conn.close()
            raise ex
        finally:
            if conn:conn.close()