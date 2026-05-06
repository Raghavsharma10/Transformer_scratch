def execute_sql(self, sql, commit=False):
        """Log and then execute a SQL query"""
        logger.info("Running sqlite query: \"%s\"", sql)
        self.connection.execute(sql)
        if commit:
            self.connection.commit()