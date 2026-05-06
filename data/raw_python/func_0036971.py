def sql(self, sql: str, *qmark_params, **named_params):
        """
        :deprecated: use self.statement to execute properly-formatted sql statements
        """
        statement = SingleSqlStatement(sql)
        return self.statement(statement).execute(*qmark_params, **named_params)