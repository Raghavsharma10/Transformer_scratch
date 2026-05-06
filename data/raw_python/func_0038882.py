def _register_service_type(cls, subclass):
        """Registers subclass handlers of various service-type-specific service
           implementations. Look for classes decorated with
           @Folder._register_service_type for hints on how this works."""
        if hasattr(subclass, '__service_type__'):
            cls._service_type_mapping[subclass.__service_type__] = subclass
            if subclass.__service_type__:
                setattr(subclass,
                        subclass.__service_type__,
                        property(lambda x: x))
        return subclass