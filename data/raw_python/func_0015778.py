def _process_query(self, query):
        """Takes a key/val pair and returns the Elasticsearch code for it"""
        key, val = query
        field_name, field_action = split_field_action(key)

        # Boost by name__action overrides boost by name.
        boost = self.field_boosts.get(key)
        if boost is None:
            boost = self.field_boosts.get(field_name)

        handler_name = 'process_query_{0}'.format(field_action)

        if field_action and hasattr(self, handler_name):
            return getattr(self, handler_name)(field_name, val, field_action)

        elif field_action in QUERY_ACTION_MAP:
            return {
                QUERY_ACTION_MAP[field_action]: _boosted_value(
                    field_name, field_action, key, val, boost)
            }

        elif field_action == 'query_string':
            # query_string has different syntax, so it's handled
            # differently.
            #
            # Note: query_string queries are not boosted with
            # .boost()---they're boosted in the query text itself.
            return {
                'query_string': {'default_field': field_name, 'query': val}
            }

        elif field_action in RANGE_ACTIONS:
            # Ranges are special and have a different syntax, so
            # we handle them separately.
            return {
                'range': {field_name: _boosted_value(
                        field_action, field_action, key, val, boost)}
           }

        elif field_action == 'range':
            lower, upper = val
            value = {
                'gte': lower,
                'lte': upper,
            }
            if boost:
                value['boost'] = boost

            return {'range': {field_name: value}}

        raise InvalidFieldActionError(
            '%s is not a valid field action' % field_action)