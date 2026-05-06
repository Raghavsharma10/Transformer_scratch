def _validate_relations(relations, services, add_error):
    """Validate relations, ensuring that the endpoints exist.

    Receive the relations and services bundle sections.
    Use the given add_error callable to register validation error.
    """
    if not relations:
        return
    for relation in relations:
        if not islist(relation):
            add_error('relation {} is malformed'.format(relation))
            continue
        relation_str = ' -> '.join('{}'.format(i) for i in relation)
        for endpoint in relation:
            if not isstring(endpoint):
                add_error(
                    'relation {} has malformed endpoint {}'
                    ''.format(relation_str, endpoint))
                continue
            try:
                service, _ = endpoint.split(':')
            except ValueError:
                service = endpoint
            if service not in services:
                add_error(
                    'relation {} endpoint {} refers to a non-existent service '
                    '{}'.format(relation_str, endpoint, service))