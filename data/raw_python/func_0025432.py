def persistent_object_context_changed(self):
        """ Override from PersistentObject. """
        super().persistent_object_context_changed()

        def detach():
            for listener in self.__interval_mutated_listeners:
                listener.close()
            self.__interval_mutated_listeners = list()

        def reattach():
            detach()
            interval_descriptors = list()
            if self.__source:
                for region in self.__source.graphics:
                    if isinstance(region, Graphics.IntervalGraphic):
                        interval_descriptor = {"interval": region.interval, "color": "#F00"}
                        interval_descriptors.append(interval_descriptor)
                        self.__interval_mutated_listeners.append(region.property_changed_event.listen(lambda k: reattach()))
            if self.__target:
                self.__target.interval_descriptors = interval_descriptors

        def item_inserted(key, value, before_index):
            if key == "graphics" and self.__target:
                reattach()

        def item_removed(key, value, index):
            if key == "graphics" and self.__target:
                reattach()

        def source_registered(source):
            self.__source = source
            self.__item_inserted_event_listener = self.__source.item_inserted_event.listen(item_inserted)
            self.__item_removed_event_listener = self.__source.item_removed_event.listen(item_removed)
            reattach()

        def target_registered(target):
            self.__target = target
            reattach()

        def unregistered(source=None):
            if self.__item_inserted_event_listener:
                self.__item_inserted_event_listener.close()
                self.__item_inserted_event_listener = None
            if self.__item_removed_event_listener:
                self.__item_removed_event_listener.close()
                self.__item_removed_event_listener = None

        if self.persistent_object_context:
            self.persistent_object_context.subscribe(self.source_uuid, source_registered, unregistered)
            self.persistent_object_context.subscribe(self.target_uuid, target_registered, unregistered)
        else:
            unregistered()