def update_table(self, unknown_type='str'):
        """Update the source table from the datafile"""
        from ambry_sources.intuit import TypeIntuiter

        st = self.source_table

        if self.reftype == 'partition':
            for c in self.partition.table.columns:
                st.add_column(c.sequence_id, source_header=c.name, dest_header=c.name,
                              datatype=c.datatype, description = c.description)

        elif self.datafile.exists:
            with self.datafile.reader as r:

                names = set()

                for col in r.columns:

                    name = col['name']

                    if name in names:  # Handle duplicate names.
                        name = name+"_"+str(col['pos'])

                    names.add(name)

                    c = st.column(name)

                    dt = col['resolved_type'] if col['resolved_type'] != 'unknown' else unknown_type

                    if c:
                        c.datatype = TypeIntuiter.promote_type(c.datatype, col['resolved_type'])

                    else:

                        c = st.add_column(col['pos'],
                                          source_header=name,
                                          dest_header=name,
                                          datatype=col['resolved_type'],
                                          description=col['description'],
                                          has_codes=col['has_codes'])