def oaiserver(sets, records):
    """Initialize OAI-PMH server."""
    from invenio_db import db
    from invenio_oaiserver.models import OAISet
    from invenio_records.api import Record

    # create a OAI Set
    with db.session.begin_nested():
        for i in range(sets):
            db.session.add(OAISet(
                spec='test{0}'.format(i),
                name='Test{0}'.format(i),
                description='test desc {0}'.format(i),
                search_pattern='title_statement.title:Test{0}'.format(i),
            ))

    # create a record
    schema = {
        'type': 'object',
        'properties': {
            'title_statement': {
                'type': 'object',
                'properties': {
                    'title': {
                        'type': 'string',
                    },
                },
            },
            'field': {'type': 'boolean'},
        },
    }

    with app.app_context():
        indexer = RecordIndexer()
        with db.session.begin_nested():
            for i in range(records):
                record_id = uuid.uuid4()
                data = {
                    'title_statement': {'title': 'Test{0}'.format(i)},
                    '$schema': schema,
                }
                recid_minter(record_id, data)
                oaiid_minter(record_id, data)
                record = Record.create(data, id_=record_id)
                indexer.index(record)

        db.session.commit()