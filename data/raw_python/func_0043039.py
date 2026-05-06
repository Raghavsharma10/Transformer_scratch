def _setop(self, query):
        """
        NO AGGREGATION, SIMPLE LIST COMPREHENSION
        """
        if isinstance(query.select, list):
            # RETURN BORING RESULT SET
            selects = FlatList()
            for s in listwrap(query.select):
                if isinstance(s.value, Mapping):
                    for k, v in s.value.items:
                        selects.append(sql_alias(v, quote_column(s.name + "." + k)))
                if isinstance(s.value, list):
                    for i, ss in enumerate(s.value):
                        selects.append(sql_alias(s.value, quote_column(s.name + "," + str(i))))
                else:
                    selects.append(sql_alias(s.value, quote_column(s.name)))

            sql = expand_template("""
                SELECT
                    {{selects}}
                FROM
                    {{table}}
                {{where}}
                {{sort}}
                {{limit}}
            """, {
                "selects": SQL(",\n".join(selects)),
                "table": self._subquery(query["from"])[0],
                "where": self._where2sql(query.where),
                "limit": self._limit2sql(query.limit),
                "sort": self._sort2sql(query.sort)
            })

            def post_process(sql):
                result = self.db.query(sql)
                for s in listwrap(query.select):
                    if isinstance(s.value, Mapping):
                        for r in result:
                            r[s.name] = {}
                            for k, v in s.value:
                                r[s.name][k] = r[s.name + "." + k]
                                r[s.name + "." + k] = None

                    if isinstance(s.value, list):
                        # REWRITE AS TUPLE
                        for r in result:
                            r[s.name] = tuple(r[s.name + "," + str(i)] for i, ss in enumerate(s.value))
                            for i, ss in enumerate(s.value):
                                r[s.name + "," + str(i)] = None

                expand_json(result)
                return result

            return sql, post_process  # RETURN BORING RESULT SET
        else:
            # RETURN LIST OF VALUES
            if query.select.value == ".":
                select = "*"
            else:
                name = query.select.name
                select = sql_alias(query.select.value, quote_column(name))

            sql = expand_template("""
                SELECT
                    {{selects}}
                FROM
                    {{table}}
                {{where}}
                {{sort}}
                {{limit}}
            """, {
                "selects": SQL(select),
                "table": self._subquery(query["from"])[0],
                "where": self._where2sql(query.where),
                "limit": self._limit2sql(query.limit),
                "sort": self._sort2sql(query.sort)
            })

            if query.select.value == ".":
                def post(sql):
                    result = self.db.query(sql)
                    expand_json(result)
                    return result

                return sql, post
            else:
                return sql, lambda sql: [r[name] for r in self.db.query(sql)]