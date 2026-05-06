def _sql(self, sql: str, params=()):
        """
        :deprecated: use self.sql instead
        """
        statement = SingleSqlStatement(sql)
        return self.statement(statement).execute_for_params(params).cursor