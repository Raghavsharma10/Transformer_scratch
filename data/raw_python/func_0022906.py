def add(self, auto_connect=None, **kwargs):
        """ Add one or more EventEmitter instances to this emitter group.
        Each keyword argument may be specified as either an EventEmitter
        instance or an Event subclass, in which case an EventEmitter will be
        generated automatically::

            # This statement:
            group.add(mouse_press=MouseEvent,
                      mouse_release=MouseEvent)

            # ..is equivalent to this statement:
            group.add(mouse_press=EventEmitter(group.source, 'mouse_press',
                                               MouseEvent),
                      mouse_release=EventEmitter(group.source, 'mouse_press',
                                                 MouseEvent))
        """
        if auto_connect is None:
            auto_connect = self.auto_connect

        # check all names before adding anything
        for name in kwargs:
            if name in self._emitters:
                raise ValueError(
                    "EmitterGroup already has an emitter named '%s'" %
                    name)
            elif hasattr(self, name):
                raise ValueError("The name '%s' cannot be used as an emitter; "
                                 "it is already an attribute of EmitterGroup"
                                 % name)

        # add each emitter specified in the keyword arguments
        for name, emitter in kwargs.items():
            if emitter is None:
                emitter = Event

            if inspect.isclass(emitter) and issubclass(emitter, Event):
                emitter = EventEmitter(
                    source=self.source,
                    type=name,
                    event_class=emitter)
            elif not isinstance(emitter, EventEmitter):
                raise Exception('Emitter must be specified as either an '
                                'EventEmitter instance or Event subclass. '
                                '(got %s=%s)' % (name, emitter))

            # give this emitter the same source as the group.
            emitter.source = self.source

            setattr(self, name, emitter)
            self._emitters[name] = emitter

            if auto_connect and self.source is not None:
                emitter.connect((self.source, self.auto_connect_format % name))

            # If emitters are connected to the group already, then this one
            # should be connected as well.
            if self._emitters_connected:
                emitter.connect(self)