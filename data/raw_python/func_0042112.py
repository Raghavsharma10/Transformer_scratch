def update(self, command):
        """
        EXPECTING command == {"set":term, "where":where}
        THE set CLAUSE IS A DICT MAPPING NAMES TO VALUES
        THE where CLAUSE IS AN ES FILTER
        """
        command = wrap(command)
        table = self.get_table(command['update'])

        es_index = self.es.cluster.get_index(read_only=False, alias=None, kwargs=self.es.settings)

        schema = table.schema

        # GET IDS OF DOCUMENTS
        query = {
            "from": command['update'],
            "select": [{"value": "_id"}] + [
                {"name": k, "value": v}
                for k, v in command.set.items()
            ],
            "where": command.where,
            "format": "list",
            "limit": 10000
        }

        results = self.query(query)

        if results.data:
            content = "".join(
                t
                for r in results.data
                for _id, row in [(r._id, r)]
                for _ in [row.__setitem__('_id', None)]  # WARNING! DESTRUCTIVE TO row
                for update in map(value2json, ({"update": {"_id": _id}}, {"doc": row}))
                for t in (update, "\n")
            )
            response = self.es.cluster.post(
                es_index.path + "/" + "_bulk",
                data=content,
                headers={"Content-Type": "application/json"},
                timeout=self.settings.timeout,
                params={"wait_for_active_shards": self.settings.wait_for_active_shards}
            )
            if response.errors:
                Log.error("could not update: {{error}}", error=[e.error for i in response["items"] for e in i.values() if e.status not in (200, 201)])

        # DELETE BY QUERY, IF NEEDED
        if "." in listwrap(command.clear):
            es_filter = self.es.cluster.lang[jx_expression(command.where)].to_esfilter(schema)
            self.es.delete_record(es_filter)
            return

        es_index.flush()