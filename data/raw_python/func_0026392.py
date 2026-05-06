def populate_user_events():
    """Generate a list of all registered authorized and anonymous events"""

    global AuthorizedEvents
    global AnonymousEvents

    def inheritors(klass):
        """Find inheritors of a specified object class"""

        subclasses = {}
        subclasses_set = set()
        work = [klass]
        while work:
            parent = work.pop()
            for child in parent.__subclasses__():
                if child not in subclasses_set:
                    # pprint(child.__dict__)
                    name = child.__module__ + "." + child.__name__
                    if name.startswith('hfos'):

                        subclasses_set.add(child)
                        event = {
                            'event': child,
                            'name': name,
                            'doc': child.__doc__,
                            'args': []
                        }

                        if child.__module__ in subclasses:
                            subclasses[child.__module__][
                                child.__name__] = event
                        else:
                            subclasses[child.__module__] = {
                                child.__name__: event
                            }
                    work.append(child)
        return subclasses

    # TODO: Change event system again, to catch authorized (i.e. "user") as
    # well as normal events, so they can be processed by Automat

    # NormalEvents = inheritors(Event)
    AuthorizedEvents = inheritors(authorizedevent)
    AnonymousEvents = inheritors(anonymousevent)