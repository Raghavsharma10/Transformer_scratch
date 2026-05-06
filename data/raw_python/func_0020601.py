def Create(event_type):
        """
        Factory method creates objects derived from :py:class`.Event` with class name matching the :py:class`.EventType`.
        
        :param event_type: number for type of event
        :returns: constructed event corresponding to ``event_type``
        :rtype: :py:class:`.Event`
        """
        if event_type in EventType.Name:
            # unknown event type gets base class
            if EventType.Name[event_type] == Event.__name__:
                return Event()
            else:
                # instantiate Event subclass with same name as EventType name
                return [t for t in EventFactory.event_list if t.__name__ == EventType.Name[event_type]][0]()
        else:
            raise TypeError("EventFactory.Create: Invalid EventType")