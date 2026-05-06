def dbcon(func):
    """Set up connection before executing function, commit and close connection
    afterwards. Unless a connection already has been created."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.dbcon is None:
            # set up connection
            self.dbcon = sqlite3.connect(self.db)
            self.dbcur = self.dbcon.cursor()
            self.dbcur.execute(SQL_SENSOR_TABLE)
            self.dbcur.execute(SQL_TMPO_TABLE)

            # execute function
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                # on exception, first close connection and then raise
                self.dbcon.rollback()
                self.dbcon.commit()
                self.dbcon.close()
                self.dbcon = None
                self.dbcur = None
                raise e
            else:
                # commit everything and close connection
                self.dbcon.commit()
                self.dbcon.close()
                self.dbcon = None
                self.dbcur = None
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper