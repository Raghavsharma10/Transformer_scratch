def serialize(pagination, **kwargs):
    """Return resumption token serializer."""
    if not pagination.has_next:
        return

    token_builder = URLSafeTimedSerializer(
        current_app.config['SECRET_KEY'],
        salt=kwargs['verb'],
    )
    schema = _schema_from_verb(kwargs['verb'], partial=False)
    data = dict(seed=random.random(), page=pagination.next_num,
                kwargs=schema.dump(kwargs).data)
    scroll_id = getattr(pagination, '_scroll_id', None)
    if scroll_id:
        data['scroll_id'] = scroll_id

    return token_builder.dumps(data)