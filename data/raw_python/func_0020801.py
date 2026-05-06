def insertSite(self, businput):
        """
        Input dictionary has to have the following keys:
        site_name
        it builds the correct dictionary for dao input and executes the dao
        """
        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            siteobj = { # FIXME: unused?
                "site_name" : businput["site_name"]
            }
            businput["site_id"] = self.sm.increment(conn, "SEQ_SI", tran)
            self.sitein.execute(conn, businput, tran)
            tran.commit()
        except Exception as ex:
            if (str(ex).lower().find("unique constraint") != -1 or
                str(ex).lower().find("duplicate") != -1):
                # already exists, lets fetch the ID
                self.logger.warning("Ignoring unique constraint violation")
                self.logger.warning(ex)
            else:
                if tran:
                    tran.rollback()
                self.logger.exception(ex)
                raise
        finally:
            if tran:
                tran.close()
            if conn:
                conn.close()