def _reload_columns(self, table_desc):
        """
        :param alias: A REAL ALIAS (OR NAME OF INDEX THAT HAS NO ALIAS)
        :return:
        """
        # FIND ALL INDEXES OF ALIAS
        es_last_updated = self.es_cluster.metatdata_last_updated

        alias = table_desc.name
        canonical_index = self.es_cluster.get_best_matching_index(alias).index
        es_metadata_update_required = not (table_desc.timestamp < es_last_updated)
        metadata = self.es_cluster.get_metadata(force=es_metadata_update_required)

        props = [
            (self.es_cluster.get_index(index=i, type=t, debug=DEBUG), t, m.properties)
            for i, d in metadata.indices.items()
            if alias in d.aliases
            for t, m in [_get_best_type_from_mapping(d.mappings)]
        ]

        # CONFIRM ALL COLUMNS ARE SAME, FIX IF NOT
        dirty = False
        all_comparisions = list(jx.pairwise(props)) + list(jx.pairwise(jx.reverse(props)))
        # NOTICE THE SAME (index, type, properties) TRIPLE FROM ABOVE
        for (i1, t1, p1), (i2, t2, p2) in all_comparisions:
            diff = elasticsearch.diff_schema(p2, p1)
            if not self.settings.read_only:
                for d in diff:
                    dirty = True
                    i1.add_property(*d)
        meta = self.es_cluster.get_metadata(force=dirty).indices[canonical_index]

        data_type, mapping = _get_best_type_from_mapping(meta.mappings)
        mapping.properties["_id"] = {"type": "string", "index": "not_analyzed"}
        columns = self._parse_properties(alias, mapping)
        table_desc.timestamp = es_last_updated
        return columns