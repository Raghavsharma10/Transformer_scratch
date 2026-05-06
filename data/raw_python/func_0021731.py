def cast_item(cls, item):
        """Cast list item to the appropriate tag type."""
        if not isinstance(item, cls.subtype):
            incompatible = isinstance(item, Base) and not any(
                issubclass(cls.subtype, tag_type) and isinstance(item, tag_type)
                for tag_type in cls.all_tags.values()
            )
            if incompatible:
                raise IncompatibleItemType(item, cls.subtype)

            try:
                return cls.subtype(item)
            except EndInstantiation:
                raise ValueError('List tags without an explicit subtype must '
                                 'either be empty or instantiated with '
                                 'elements from which a subtype can be '
                                 'inferred') from None
            except (IncompatibleItemType, CastError):
                raise
            except Exception as exc:
                raise CastError(item, cls.subtype) from exc
        return item