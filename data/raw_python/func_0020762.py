def insertOutputConfig(self, businput):
        """
        Method to insert the Output Config.
        app_name, release_version, pset_hash, global_tag and output_module_label are
        required.
        args:
            businput(dic): input dictionary. 

        Updated Oct 12, 2011    
        """
        if not ("app_name" in businput  and "release_version" in businput\
            and "pset_hash" in businput and "output_module_label" in businput
            and "global_tag" in businput):
            dbsExceptionHandler('dbsException-invalid-input', "business/DBSOutputConfig/insertOutputConfig require:\
                app_name, release_version, pset_hash, output_module_label and global_tag")

        conn = self.dbi.connection()
        tran = conn.begin()
        try:
            # Proceed with o/p module insertion
            businput['scenario'] = businput.get("scenario", None)
            businput['pset_name'] = businput.get("pset_name", None)
            self.outmodin.execute(conn, businput, tran)
            tran.commit()
            tran = None
        except SQLAlchemyIntegrityError as ex:
            if str(ex).find("unique constraint") != -1 or str(ex).lower().find("duplicate") != -1:
                #if the validation is due to a unique constrain break in OUTPUT_MODULE_CONFIGS
                if str(ex).find("TUC_OMC_1") != -1: pass
                #otherwise, try again
                else:
                    try:
                        self.outmodin.execute(conn, businput, tran)
                        tran.commit()
                        tran =  None
                    except SQLAlchemyIntegrityError as ex1:
                        if str(ex1).find("unique constraint") != -1 and str(ex1).find("TUC_OMC_1") != -1: pass
                    except Exception as e1:
                        if tran:
                            tran.rollback()
                            tran = None
                        raise
            else:
                raise
        except Exception as e:
            if tran:
                tran.rollback()
            raise
        finally:
            if tran:
                tran.rollback()
            if conn:
                conn.close()