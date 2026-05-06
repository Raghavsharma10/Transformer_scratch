def _compile_query(self):
        """
        Builds SOLR query and stores it into self.compiled_query
        """
        # https://wiki.apache.org/solr/SolrQuerySyntax
        # http://lucene.apache.org/core/2_9_4/queryparsersyntax.html
        query = []

        # filtered_query = self._model_class.row_level_access(self._current_context, self)
        # if filtered_query is not None:
        #     self._solr_query += filtered_query._solr_query
        # print(self._solr_query)
        for key, val, is_escaped in self._solr_query:
            # querying on a linked model by model instance
            # it should be a Model, not a Node!
            if key == 'key':
                key = '_yz_rk'
            elif key == '-key':
                    key = '-_yz_rk'
            elif key[:5] == 'key__':  # to handle key__in etc.
                key = '_yz_rk__' + key[5:]
            elif key[:6] == '-key__':  # to handle key__in etc.
                key = '-_yz_rk__' + key[6:]

            key, val, is_escaped = self._process_query_val(key, val, is_escaped)
            # if it's not one of the expected objects, it should be a string
            # if key == "OR_QRY" then join them with "OR" after escaping & parsing
            if key == 'OR_QRY':
                key = 'NOKEY'
                val = ' OR '.join(
                    ['%s:%s' % self._parse_query_key(*self._process_query_val(k, v, is_escaped)) for
                     k, v in val.items()])
                is_escaped = True
            # __in query is same as OR_QRY but key stays same for all values
            elif key.endswith('__in'):
                if not val:
                    raise ValueError("query value list can not be empty for __in query, "
                                     "please check if it is empty or not before execute filter.")
                key = key[:-4]
                val = ' OR '.join(
                    ['%s:%s' % (key, self._escape_query(v, is_escaped)) for v in val])
                if key.startswith('-'):
                    val = '*:* %s' % val
                key = 'NOKEY'
                is_escaped = True
            # parse the query
            key, val = self._parse_query_key(key, val, is_escaped)

            # as long as not explicitly asked for,
            # we filter out records with deleted flag
            if key == 'deleted':
                self.want_deleted = True
            # convert two underscores to dot notation
            key = key.replace('__', '.')
            # NOKEY means we already combined key partition in to "val"
            if key == 'NOKEY':
                query.append("(%s)" % val)
            else:
                query.append("%s:%s" % (key, val))

        # need to add *:* for negative queries, if
        # query has only one criteria, such as:
        # (-name:Jack) AND -deleted:True
        # this wont work properly, it must be altered as
        # (*:* -name:Jack) AND -deleted:True
        if len(query) == 1:
            q = query[0]
            if q.startswith('-'):
                query[0] = '*:* %s' % q
            if q[:2] == '(-':
                query[0] = '( *:* %s' % q[1:]

        # filter out "deleted" fields if not user explicitly asked for

        # join everything with "AND"
        joined_query = self._QUERY_GLUE.join(query)
        if not self.want_deleted:
            if joined_query:
                joined_query = "(%s) AND -deleted:True" % joined_query
            else:
                joined_query = '-deleted:True'
        elif not joined_query:
            joined_query = '*:*'
        self.compiled_query = joined_query