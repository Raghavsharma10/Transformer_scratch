def _update_cardinality(self, column):
        """
        QUERY ES TO FIND CARDINALITY AND PARTITIONS FOR A SIMPLE COLUMN
        """
        now = Date.now()
        if column.es_index in self.index_does_not_exist:
            return

        if column.jx_type in STRUCT:
            Log.error("not supported")
        try:
            if column.es_index == "meta.columns":
                partitions = jx.sort([g[column.es_column] for g, _ in jx.groupby(self.meta.columns, column.es_column) if g[column.es_column] != None])
                self.meta.columns.update({
                    "set": {
                        "partitions": partitions,
                        "count": len(self.meta.columns),
                        "cardinality": len(partitions),
                        "multi": 1,
                        "last_updated": now
                    },
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            if column.es_index == "meta.tables":
                partitions = jx.sort([g[column.es_column] for g, _ in jx.groupby(self.meta.tables, column.es_column) if g[column.es_column] != None])
                self.meta.columns.update({
                    "set": {
                        "partitions": partitions,
                        "count": len(self.meta.tables),
                        "cardinality": len(partitions),
                        "multi": 1,
                        "last_updated": now
                    },
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return

            es_index = column.es_index.split(".")[0]

            is_text = [cc for cc in self.meta.columns if cc.es_column == column.es_column and cc.es_type == "text"]
            if is_text:
                # text IS A MULTIVALUE STRING THAT CAN ONLY BE FILTERED
                result = self.es_cluster.post("/" + es_index + "/_search", data={
                    "aggs": {
                        "count": {"filter": {"match_all": {}}}
                    },
                    "size": 0
                })
                count = result.hits.total
                cardinality = max(1001, count)
                multi = 1001
            elif column.es_column == "_id":
                result = self.es_cluster.post("/" + es_index + "/_search", data={
                    "query": {"match_all": {}},
                    "size": 0
                })
                count = cardinality = result.hits.total
                multi = 1
            elif column.es_type == BOOLEAN:
                result = self.es_cluster.post("/" + es_index + "/_search", data={
                    "aggs": {
                        "count": _counting_query(column)
                    },
                    "size": 0
                })
                count = result.hits.total
                cardinality = 2

                DEBUG and Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "partitions": [False, True],
                        "multi": 1,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            else:
                es_query = {
                    "aggs": {
                        "count": _counting_query(column),
                        "_filter": {
                            "aggs": {"multi": {"max": {"script": "doc[" + quote(column.es_column) + "].values.size()"}}},
                            "filter": {"bool": {"should": [
                                {"range": {"etl.timestamp.~n~": {"gte": (Date.today() - WEEK)}}},
                                {"bool": {"must_not": {"exists": {"field": "etl.timestamp.~n~"}}}}
                            ]}}
                        }
                    },
                    "size": 0
                }

                result = self.es_cluster.post("/" + es_index + "/_search", data=es_query)
                agg_results = result.aggregations
                count = result.hits.total
                cardinality = coalesce(agg_results.count.value, agg_results.count._nested.value, agg_results.count.doc_count)
                multi = int(coalesce(agg_results._filter.multi.value, 1))
                if cardinality == None:
                    Log.error("logic error")

            query = Data(size=0)

            if column.es_column == "_id":
                self.meta.columns.update({
                    "set": {
                        "count": cardinality,
                        "cardinality": cardinality,
                        "multi": 1,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            elif cardinality > 1000 or (count >= 30 and cardinality == count) or (count >= 1000 and cardinality / count > 0.99):
                DEBUG and Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "multi": multi,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            elif column.es_type in elasticsearch.ES_NUMERIC_TYPES and cardinality > 30:
                DEBUG and Log.note("{{table}}.{{field}} has {{num}} parts", table=column.es_index, field=column.es_column, num=cardinality)
                self.meta.columns.update({
                    "set": {
                        "count": count,
                        "cardinality": cardinality,
                        "multi": multi,
                        "last_updated": now
                    },
                    "clear": ["partitions"],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                return
            elif len(column.nested_path) != 1:
                query.aggs["_"] = {
                    "nested": {"path": column.nested_path[0]},
                    "aggs": {"_nested": {"terms": {"field": column.es_column}}}
                }
            elif cardinality == 0:  # WHEN DOES THIS HAPPEN?
                query.aggs["_"] = {"terms": {"field": column.es_column}}
            else:
                query.aggs["_"] = {"terms": {"field": column.es_column, "size": cardinality}}

            result = self.es_cluster.post("/" + es_index + "/_search", data=query)

            aggs = result.aggregations._
            if aggs._nested:
                parts = jx.sort(aggs._nested.buckets.key)
            else:
                parts = jx.sort(aggs.buckets.key)

            DEBUG and Log.note("update metadata for {{column.es_index}}.{{column.es_column}} (id={{id}}) at {{time}}", id=id(column), column=column, time=now)
            self.meta.columns.update({
                "set": {
                    "count": count,
                    "cardinality": cardinality,
                    "multi": multi,
                    "partitions": parts,
                    "last_updated": now
                },
                "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
            })
        except Exception as e:
            # CAN NOT IMPORT: THE TEST MODULES SETS UP LOGGING
            # from tests.test_jx import TEST_TABLE
            e = Except.wrap(e)
            TEST_TABLE = "testdata"
            is_missing_index = any(w in e for w in ["IndexMissingException", "index_not_found_exception"])
            is_test_table = column.es_index.startswith((TEST_TABLE_PREFIX, TEST_TABLE))
            if is_missing_index:
                # WE EXPECT TEST TABLES TO DISAPPEAR
                Log.warning("Missing index {{col.es_index}}", col=column, cause=e)
                self.meta.columns.update({
                    "clear": ".",
                    "where": {"eq": {"es_index": column.es_index}}
                })
                self.index_does_not_exist.add(column.es_index)
            elif "No field found for" in e:
                self.meta.columns.update({
                    "clear": ".",
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                Log.warning("Could not get column {{col.es_index}}.{{col.es_column}} info", col=column, cause=e)
            else:
                self.meta.columns.update({
                    "set": {
                        "last_updated": now
                    },
                    "clear": [
                        "count",
                        "cardinality",
                        "multi",
                        "partitions",
                    ],
                    "where": {"eq": {"es_index": column.es_index, "es_column": column.es_column}}
                })
                Log.warning("Could not get {{col.es_index}}.{{col.es_column}} info", col=column, cause=e)