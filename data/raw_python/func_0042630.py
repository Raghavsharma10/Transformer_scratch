def create_event(self, actors=None, ignore_duplicates=False, **kwargs):
        """
        Create events with actors.

        This method can be used in place of ``Event.objects.create``
        to create events, and the appropriate actors. It takes all the
        same keywords as ``Event.objects.create`` for the event
        creation, but additionally takes a list of actors, and can be
        told to not attempt to create an event if a duplicate event
        exists.

        :type source: Source
        :param source: A ``Source`` object representing where the
            event came from.

        :type context: dict
        :param context: A dictionary containing relevant
            information about the event, to be serialized into
            JSON. It is possible to load additional context
            dynamically  when events are fetched. See the
            documentation on the ``ContextRenderer`` model.

        :type uuid: str
        :param uuid: A unique string for the event. Requiring a
            ``uuid`` allows code that creates events to ensure they do
            not create duplicate events. This id could be, for example
            some hash of the ``context``, or, if the creator is
            unconcerned with creating duplicate events a call to
            python's ``uuid1()`` in the ``uuid`` module.

        :type time_expires: datetime (optional)
        :param time_expires: If given, the default methods for
            querying events will not return this event after this time
            has passed.

        :type actors: (optional) List of entities or list of entity ids.
        :param actors: An ``EventActor`` object will be created for
            each entity in the list. This allows for subscriptions
            which are only following certain entities to behave
            appropriately.

        :type ignore_duplicates: (optional) Boolean
        :param ignore_duplicates: If ``True``, a check will be made to
            ensure that an event with the give ``uuid`` does not exist
            before attempting to create the event. Setting this to
            ``True`` allows the creator of events to gracefully ensure
            no duplicates are attempted to be created. There is a uniqueness constraint on uuid
            so it will raise an exception if duplicates are allowed and submitted.

        :rtype: Event
        :returns: The created event. Alternatively if a duplicate
            event already exists and ``ignore_duplicates`` is
            ``True``, it will return ``None``.
        """
        kwargs['actors'] = actors
        kwargs['ignore_duplicates'] = ignore_duplicates

        events = self.create_events([kwargs])

        if events:
            return events[0]

        return None