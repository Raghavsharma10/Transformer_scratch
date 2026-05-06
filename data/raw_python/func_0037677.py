def _save_percolator(self):
        """
        Saves the query field as an elasticsearch percolator
        """
        index = Content.search_objects.mapping.index
        query_filter = self.get_content(published=False).to_dict()

        q = {}

        if "query" in query_filter:
            q = {"query": query_filter.get("query", {})}
        else:
            # We don't know how to save this
            return

        # We'll need this data, to decide which special coverage section to use
        q["sponsored"] = bool(self.tunic_campaign_id)
        # Elasticsearch v1.4 percolator "field_value_factor" does not
        # support missing fields, so always need to include

        q["start_date"] = self.start_date
        # NOTE: set end_date to datetime.max if special coverage has no end date
        # (i.e. is a neverending special coverage)
        q["end_date"] = self.end_date if self.end_date else datetime.max.replace(tzinfo=pytz.UTC)

        # Elasticsearch v1.4 percolator range query does not support DateTime range queries
        # (PercolateContext.nowInMillisImpl is not implemented).
        if q["start_date"]:
            q['start_date_epoch'] = datetime_to_epoch_seconds(q["start_date"])
        if q["end_date"]:
            q['end_date_epoch'] = datetime_to_epoch_seconds(q["end_date"])

        # Store manually included IDs for percolator retrieval scoring (boost
        # manually included content).
        if self.query:
            q['included_ids'] = self.query.get('included_ids', [])

        es.index(
            index=index,
            doc_type=".percolator",
            body=q,
            id=self.es_id
        )