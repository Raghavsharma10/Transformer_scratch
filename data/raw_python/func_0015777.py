def _process_filters(self, filters):
        """Takes a list of filters and returns ES JSON API

        :arg filters: list of F, (key, val) tuples, or dicts

        :returns: list of ES JSON API filters

        """
        rv = []
        for f in filters:
            if isinstance(f, F):
                if f.filters:
                    rv.extend(self._process_filters(f.filters))
                    continue

            elif isinstance(f, dict):
                if six.PY3:
                    key = list(f.keys())[0]
                else:
                    key = f.keys()[0]
                val = f[key]
                key = key.strip('_')

                if key not in ('or', 'and', 'not', 'filter'):
                    raise InvalidFieldActionError(
                        '%s is not a valid connector' % f.keys()[0])

                if 'filter' in val:
                    filter_filters = self._process_filters(val['filter'])
                    if len(filter_filters) == 1:
                        filter_filters = filter_filters[0]
                    rv.append({key: {'filter': filter_filters}})
                else:
                    rv.append({key: self._process_filters(val)})

            else:
                key, val = f
                key, field_action = split_field_action(key)
                handler_name = 'process_filter_{0}'.format(field_action)

                if field_action and hasattr(self, handler_name):
                    rv.append(getattr(self, handler_name)(
                            key, val, field_action))

                elif key.strip('_') in ('or', 'and', 'not'):
                    connector = key.strip('_')
                    rv.append({connector: self._process_filters(val.items())})

                elif field_action is None:
                    if val is None:
                        rv.append({'missing': {
                                    'field': key, "null_value": True}})
                    else:
                        rv.append({'term': {key: val}})

                elif field_action in ('startswith', 'prefix'):
                    rv.append({'prefix': {key: val}})

                elif field_action == 'in':
                    rv.append({'in': {key: val}})

                elif field_action in RANGE_ACTIONS:
                    rv.append({'range': {key: {field_action: val}}})

                elif field_action == 'range':
                    lower, upper = val
                    rv.append({'range': {key: {'gte': lower, 'lte': upper}}})

                elif field_action == 'distance':
                    distance, latitude, longitude = val

                    rv.append({
                        'geo_distance': {
                            'distance': distance,
                            key: [longitude, latitude]
                        }
                    })

                else:
                    raise InvalidFieldActionError(
                        '%s is not a valid field action' % field_action)

        return rv