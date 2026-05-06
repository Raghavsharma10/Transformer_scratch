def _as_document(self, partition):
        """ Converts given partition to the document indexed by FTS backend.

        Args:
            partition (orm.Partition): partition to convert.

        Returns:
            dict with structure matches to BasePartitionIndex._schema.

        """

        schema = ' '.join(
            u'{} {} {} {} {}'.format(
                c.id,
                c.vid,
                c.name,
                c.altname,
                c.description) for c in partition.table.columns)

        values = ''

        for stat in partition.stats:
            if stat.uvalues :
                # SOme geometry vlaues are super long. They should not be in uvbalues, but when they are,
                # need to cut them down.
                values += ' '.join(e[:200] for e in stat.uvalues) + '\n'

        # Re-calculate the summarization of grains, since the geoid 0.0.7 package had a bug where state level
        # summaries had the same value as state-level allvals
        def resum(g):
            try:
                return str(GVid.parse(g).summarize())
            except KeyError:
                return g
            except ValueError:
                logger.debug("Failed to parse gvid '{}' from partition '{}' grain coverage"
                             .format(g, partition.identity.vname))
                return g

        keywords = (
            ' '.join(partition.space_coverage) + ' ' +
            ' '.join([resum(g) for g in partition.grain_coverage if resum(g)]) + ' ' +
            ' '.join(str(x) for x in partition.time_coverage)
        )

        doc_field = u('{} {} {} {} {} {}').format(
            values,
            schema,
            ' '.join([
                u('{}').format(partition.identity.vid),
                u('{}').format(partition.identity.id_),
                u('{}').format(partition.identity.name),
                u('{}').format(partition.identity.vname)]),
            partition.display.title,
            partition.display.description,
            partition.display.sub_description,
            partition.display.time_description,
            partition.display.geo_description
        )

        document = dict(
            vid=u('{}').format(partition.identity.vid),
            dataset_vid=u('{}').format(partition.identity.as_dataset().vid),
            title=u('{}').format(partition.table.description),
            keywords=u('{}').format(keywords),
            doc=doc_field)

        return document