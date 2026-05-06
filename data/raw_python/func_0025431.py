def persistent_object_context_changed(self):
        """ Override from PersistentObject. """
        super().persistent_object_context_changed()

        def register():
            if self.__source is not None and self.__target is not None:
                assert not self.__binding
                self.__binding = Binding.PropertyBinding(self.__source, self.source_property)
                self.__binding.target_setter = self.__set_target_from_source
                # while reading, the data item in the display data channel will not be connected;
                # we still set its value here. when the data item becomes valid, it will update.
                self.__binding.update_target_direct(self.__binding.get_target_value())

        def source_registered(source):
            self.__source = source
            register()

        def target_registered(target):
            self.__target = target

            def property_changed(target, property_name):
                if property_name == self.target_property:
                    self.__set_source_from_target(getattr(target, property_name))

            assert self.__target_property_changed_listener is None
            self.__target_property_changed_listener = target.property_changed_event.listen(functools.partial(property_changed, target))
            register()

        def unregistered(item=None):
            if not item or item == self.__source:
                self.__source = None
            if not item or item == self.__target:
                self.__target = None
            if self.__binding:
                self.__binding.close()
                self.__binding = None
            if self.__target_property_changed_listener:
                self.__target_property_changed_listener.close()
                self.__target_property_changed_listener = None

        def change_registration(registered_object, unregistered_object):
            if registered_object and registered_object.uuid == self.source_uuid:
                source_registered(registered_object)
            if registered_object and registered_object.uuid == self.target_uuid:
                target_registered(registered_object)
            if unregistered_object and unregistered_object in (self._source, self._target):
                unregistered(unregistered_object)

        if self.persistent_object_context:
            self.__registration_listener = self.persistent_object_context.registration_event.listen(change_registration)
            source = self.persistent_object_context.get_registered_object(self.source_uuid)
            target = self.persistent_object_context.get_registered_object(self.target_uuid)
            if source:
                source_registered(source)
            if target:
                target_registered(target)
        else:
            unregistered()