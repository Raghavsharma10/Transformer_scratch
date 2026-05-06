def stream(self, model, position):
        """Create a :class:`~bloop.stream.Stream` that provides approximate chronological ordering.

        .. code-block:: pycon

            # Create a user so we have a record
            >>> engine = Engine()
            >>> user = User(id=3, email="user@domain.com")
            >>> engine.save(user)
            >>> user.email = "admin@domain.com"
            >>> engine.save(user)

            # First record lacks an "old" value since it's an insert
            >>> stream = engine.stream(User, "trim_horizon")
            >>> next(stream)
            {'key': None,
             'old': None,
             'new': User(email='user@domain.com', id=3, verified=None),
             'meta': {
                 'created_at': datetime.datetime(2016, 10, 23, ...),
                 'event': {
                     'id': '3fe6d339b7cb19a1474b3d853972c12a',
                     'type': 'insert',
                     'version': '1.1'},
                 'sequence_number': '700000000007366876916'}
            }


        :param model: The model to stream records from.
        :param position: "trim_horizon", "latest", a stream token, or a :class:`datetime.datetime`.
        :return: An iterator for records in all shards.
        :rtype: :class:`~bloop.stream.Stream`
        :raises bloop.exceptions.InvalidStream: if the model does not have a stream.
        """
        validate_not_abstract(model)
        if not model.Meta.stream or not model.Meta.stream.get("arn"):
            raise InvalidStream("{!r} does not have a stream arn".format(model))
        stream = Stream(model=model, engine=self)
        stream.move_to(position=position)
        return stream