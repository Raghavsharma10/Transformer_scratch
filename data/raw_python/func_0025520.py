def persistent_object_context_changed(self):
        """ Override from PersistentObject. """
        super().persistent_object_context_changed()

        def source_registered(source):
            self.__source = source

        def source_unregistered(source=None):
            pass

        def reference_registered(property_name, reference):
            self.__referenced_objects[property_name] = reference

        def reference_unregistered(property_name, reference=None):
            pass

        if self.persistent_object_context:
            self.persistent_object_context.subscribe(self.source_uuid, source_registered, source_unregistered)

            for property_name, value in self.__properties.items():
                if isinstance(value, dict) and value.get("type") in {"data_item", "display_item", "data_source", "graphic", "structure"} and "uuid" in value:
                    self.persistent_object_context.subscribe(uuid.UUID(value["uuid"]), functools.partial(reference_registered, property_name), functools.partial(reference_unregistered, property_name))
        else:
            source_unregistered()

            for property_name, value in self.__properties.items():
                if isinstance(value, dict) and value.get("type") in {"data_item", "display_item", "data_source", "graphic", "structure"} and "uuid" in value:
                    reference_unregistered(property_name)