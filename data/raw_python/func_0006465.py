def _clean_doc(self, doc, namespace, timestamp):
        """Reformats the given document before insertion into Solr.

        This method reformats the document in the following ways:
          - removes extraneous fields that aren't defined in schema.xml
          - unwinds arrays in order to find and later flatten sub-documents
          - flattens the document so that there are no sub-documents, and every
            value is associated with its dot-separated path of keys
          - inserts namespace and timestamp metadata into the document in order
            to handle rollbacks

        An example:
          {"a": 2,
           "b": {
             "c": {
               "d": 5
             }
           },
           "e": [6, 7, 8]
          }

        becomes:
          {"a": 2, "b.c.d": 5, "e.0": 6, "e.1": 7, "e.2": 8}

        """

        # Translate the _id field to whatever unique key we're using.
        # _id may not exist in the doc, if we retrieved it from Solr
        # as part of update.
        if '_id' in doc:
            doc[self.unique_key] = u(doc.pop("_id"))

        # Update namespace and timestamp metadata
        if 'ns' in doc or '_ts' in doc:
            raise errors.OperationFailed(
                'Need to set "ns" and "_ts" fields, but these fields already '
                'exist in the document %r!' % doc)
        doc['ns'] = namespace
        doc['_ts'] = timestamp

        # SOLR cannot index fields within sub-documents, so flatten documents
        # with the dot-separated path to each value as the respective key
        flat_doc = self._formatter.format_document(doc)

        # Only include fields that are explicitly provided in the
        # schema or match one of the dynamic field patterns, if
        # we were able to retrieve the schema
        if len(self.field_list) + len(self._dynamic_field_regexes) > 0:
            def include_field(field):
                return field in self.field_list or any(
                    regex.match(field) for regex in self._dynamic_field_regexes
                )
            return dict((k, v) for k, v in flat_doc.items() if include_field(k))
        return flat_doc