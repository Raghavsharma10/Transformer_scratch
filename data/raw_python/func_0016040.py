def timestamp(instance):
    """Ensure timestamps contain sane months, days, hours, minutes, seconds.
    """
    ts_re = re.compile(r"^[0-9]{4}-(0[1-9]|1[012])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9]|60)(\.[0-9]+)?Z$")
    timestamp_props = ['created', 'modified']
    if instance['type'] in enums.TIMESTAMP_PROPERTIES:
        timestamp_props += enums.TIMESTAMP_PROPERTIES[instance['type']]

    for tprop in timestamp_props:
        if tprop in instance and ts_re.match(instance[tprop]):
            # Don't raise an error if schemas will catch it
            try:
                parser.parse(instance[tprop])
            except ValueError as e:
                yield JSONError("'%s': '%s' is not a valid timestamp: %s"
                                % (tprop, instance[tprop], str(e)), instance['id'])

    if has_cyber_observable_data(instance):
        for key, obj in instance['objects'].items():
            if 'type' not in obj:
                continue
            if obj['type'] in enums.TIMESTAMP_OBSERVABLE_PROPERTIES:
                for tprop in enums.TIMESTAMP_OBSERVABLE_PROPERTIES[obj['type']]:
                    if tprop in obj and ts_re.match(obj[tprop]):
                        # Don't raise an error if schemas will catch it
                        try:
                            parser.parse(obj[tprop])
                        except ValueError as e:
                            yield JSONError("'%s': '%s': '%s' is not a valid timestamp: %s"
                                            % (obj['type'], tprop, obj[tprop], str(e)), instance['id'])
            if obj['type'] in enums.TIMESTAMP_EMBEDDED_PROPERTIES:
                for embed in enums.TIMESTAMP_EMBEDDED_PROPERTIES[obj['type']]:
                    if embed in obj:
                        for tprop in enums.TIMESTAMP_EMBEDDED_PROPERTIES[obj['type']][embed]:
                            if embed == 'extensions':
                                for ext in obj[embed]:
                                    if tprop in obj[embed][ext] and ts_re.match(obj[embed][ext][tprop]):
                                        try:
                                            parser.parse(obj[embed][ext][tprop])
                                        except ValueError as e:
                                            yield JSONError("'%s': '%s': '%s': '%s' is not a valid timestamp: %s"
                                                            % (obj['type'], ext, tprop, obj[embed][ext][tprop], str(e)), instance['id'])
                            elif tprop in obj[embed] and ts_re.match(obj[embed][tprop]):
                                try:
                                    parser.parse(obj[embed][tprop])
                                except ValueError as e:
                                    yield JSONError("'%s': '%s': '%s' is not a valid timestamp: %s"
                                                    % (obj['type'], tprop, obj[embed][tprop], str(e)), instance['id'])