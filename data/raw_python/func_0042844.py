def get_columns(self, table_name, column_name=None, after=None, timeout=None):
        """
        RETURN METADATA COLUMNS

        :param table_name: TABLE WE WANT COLUMNS FOR
        :param column_name:  OPTIONAL NAME, IF INTERESTED IN ONLY ONE COLUMN
        :param after: FORCE LOAD, WAITING FOR last_updated TO BE AFTER THIS TIME
        :param timeout: Signal; True when should give up
        :return:
        """
        DEBUG and after and Log.note("getting columns for after {{time}}", time=after)
        table_path = split_field(table_name)
        root_table_name = table_path[0]

        alias = self._find_alias(root_table_name)
        if not alias:
            self.es_cluster.get_metadata(force=True)
            alias = self._find_alias(root_table_name)
            if not alias:
                Log.error("{{table|quote}} does not exist", table=table_name)

        try:
            table = self.get_table(alias)[0]
            # LAST TIME WE GOT INFO FOR THIS TABLE
            if not table:
                table = TableDesc(
                    name=alias,
                    url=None,
                    query_path=["."],
                    timestamp=Date.MIN
                )
                with self.meta.tables.locker:
                    self.meta.tables.add(table)
                columns = self._reload_columns(table)
                DEBUG and Log.note("columns from reload")
            elif after or table.timestamp < self.es_cluster.metatdata_last_updated:
                columns = self._reload_columns(table)
                DEBUG and Log.note("columns from reload")
            else:
                columns = self.meta.columns.find(alias, column_name)
                DEBUG and Log.note("columns from find()")

            DEBUG and Log.note("columns are {{ids}}", ids=[id(c) for c in columns])

            columns = jx.sort(columns, "name")

            if after is None:
                return columns  # DO NOT WAIT FOR COMPLETE COLUMNS

            # WAIT FOR THE COLUMNS TO UPDATE
            while True:
                pending = [c for c in columns if after >= c.last_updated or (c.cardinality == None and c.jx_type not in STRUCT)]
                if not pending:
                    break
                if timeout:
                    Log.error("trying to gets columns timed out")
                if DEBUG:
                    if len(pending) > 10:
                        Log.note("waiting for {{num}} columns to update by {{timestamp}}", num=len(pending), timestamp=after)
                    else:
                        Log.note("waiting for columns to update by {{timestamp}}; {{columns|json}}", timestamp=after, columns=[c.es_index + "." + c.es_column + " id="+text_type(id(c)) for c in pending])
                Till(seconds=1).wait()
            return columns
        except Exception as e:
            Log.error("Failure to get columns for {{table}}", table=table_name, cause=e)

        return []