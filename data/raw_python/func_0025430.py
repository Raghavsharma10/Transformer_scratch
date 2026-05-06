def persistent_object_context_changed(self):
        """ Override from PersistentObject. """
        super().persistent_object_context_changed()

        def change_registration(registered_object, unregistered_object):
            if registered_object and registered_object.uuid == self.parent_uuid:
                self.__parent = registered_object

        if self.persistent_object_context:
            self.__registration_listener = self.persistent_object_context.registration_event.listen(change_registration)

            self.__parent = self.persistent_object_context.get_registered_object(self.parent_uuid)