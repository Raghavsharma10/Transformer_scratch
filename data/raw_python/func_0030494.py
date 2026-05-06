def _as_document(self, dataset):
        """ Converts dataset to document indexed by to FTS index.

        Args:
            dataset (orm.Dataset): dataset to convert.

        Returns:
            dict with structure matches to BaseDatasetIndex._schema.

        """

        # find tables.

        assert isinstance(dataset, Dataset)

        execute = object_session(dataset).connection().execute

        query = text("""
            SELECT t_name, c_name, c_description
            FROM columns
            JOIN tables ON c_t_vid = t_vid WHERE t_d_vid = :dataset_vid;""")

        columns = u('\n').join(
            [u(' ').join(list(text_type(e) for e in t)) for t in execute(query, dataset_vid=str(dataset.identity.vid))])

        doc = '\n'.join([u('{}').format(x) for x in [dataset.config.metadata.about.title,
                                                     dataset.config.metadata.about.summary,
                                                     dataset.identity.id_,
                                                     dataset.identity.vid,
                                                     dataset.identity.source,
                                                     dataset.identity.name,
                                                     dataset.identity.vname,
                                                     columns]])

        # From the source, make a variety of combinations for keywords:
        # foo.bar.com -> "foo foo.bar foo.bar.com bar.com"
        parts = u('{}').format(dataset.identity.source).split('.')
        sources = (['.'.join(g) for g in [parts[-i:] for i in range(2, len(parts) + 1)]]
                   + ['.'.join(g) for g in [parts[:i] for i in range(0, len(parts))]])

        # Re-calculate the summarization of grains, since the geoid 0.0.7 package had a bug where state level
        # summaries had the same value as state-level allvals
        def resum(g):
            try:
                return str(GVid.parse(g).summarize())
            except (KeyError, ValueError):
                return g

        def as_list(value):
            """ Converts value to the list. """
            if not value:
                return []
            if isinstance(value, string_types):
                lst = [value]
            else:
                try:
                    lst = list(value)
                except TypeError:
                    lst = [value]
            return lst

        about_time = as_list(dataset.config.metadata.about.time)
        about_grain = as_list(dataset.config.metadata.about.grain)

        keywords = (
            list(dataset.config.metadata.about.groups) +
            list(dataset.config.metadata.about.tags) +
            about_time +
            [resum(g) for g in about_grain] +
            sources)

        document = dict(
            vid=u('{}').format(dataset.identity.vid),
            title=u('{} {}').format(dataset.identity.name, dataset.config.metadata.about.title),
            doc=u('{}').format(doc),
            keywords=' '.join(u('{}').format(x) for x in keywords)
        )

        return document