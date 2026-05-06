def index(advice):
    '''Index/Reindex a CADA advice'''
    topics = []
    for topic in advice.topics:
        topics.append(topic)
        parts = topic.split('/')
        if len(parts) > 1:
            topics.append(parts[0])

    try:
        es.index(index=es.index_name, doc_type=DOCTYPE, id=advice.id, body={
            'id': advice.id,
            'administration': advice.administration,
            'type': advice.type,
            'session': advice.session.strftime('%Y-%m-%d'),
            'subject': advice.subject,
            'topics': topics,
            'tags': advice.tags,
            'meanings': advice.meanings,
            'part': advice.part,
            'content': advice.content,
        })
    except Exception:
        log.exception('Unable to index advice %s', advice.id)