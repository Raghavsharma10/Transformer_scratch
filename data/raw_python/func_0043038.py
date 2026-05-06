def _aggop(self, query):
        """
        SINGLE ROW RETURNED WITH AGGREGATES
        """
        if isinstance(query.select, list):
            # RETURN SINGLE OBJECT WITH AGGREGATES
            for s in query.select:
                if s.aggregate not in aggregates:
                    Log.error("Expecting all columns to have an aggregate: {{select}}", select=s)

            selects = FlatList()
            for s in query.select:
                selects.append(sql_alias(aggregates[s.aggregate].replace("{{code}}", s.value),quote_column(s.name)))

            sql = expand_template("""
                SELECT
                    {{selects}}
                FROM
                    {{table}}
                {{where}}
            """, {
                "selects": SQL(",\n".join(selects)),
                "table": self._subquery(query["from"])[0],
                "where": self._where2sql(query.filter)
            })

            return sql, lambda sql: self.db.column(sql)[0]  # RETURNING SINGLE OBJECT WITH AGGREGATE VALUES
        else:
            # RETURN SINGLE VALUE
            s0 = query.select
            if s0.aggregate not in aggregates:
                Log.error("Expecting all columns to have an aggregate: {{select}}", select=s0)

            select = sql_alias(aggregates[s0.aggregate].replace("{{code}}", s0.value) , quote_column(s0.name))

            sql = expand_template("""
                SELECT
                    {{selects}}
                FROM
                    {{table}}
                {{where}}
            """, {
                "selects": SQL(select),
                "table": self._subquery(query["from"])[0],
                "where": self._where2sql(query.where)
            })

            def post(sql):
                result = self.db.column_query(sql)
                return result[0][0]

            return sql, post