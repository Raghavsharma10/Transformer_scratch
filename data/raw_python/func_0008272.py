def get_value(self, Meta: Type[object], base_classes_meta, mcs_args: McsArgs) -> Any:
        """
        Returns the value for ``self.name`` given the class-under-construction's class
        ``Meta``. If it's not found there, and ``self.inherit == True`` and there is a
        base class that has a class ``Meta``, use that value, otherwise ``self.default``.

        :param Meta: the class ``Meta`` (if any) from the class-under-construction
                     (**NOTE:** this will be an ``object`` or ``None``, NOT an instance
                     of :class:`MetaOptionsFactory`)
        :param base_classes_meta: the :class:`MetaOptionsFactory` instance (if any) from
                                  the base class of the class-under-construction
        :param mcs_args: the :class:`McsArgs` for the class-under-construction
        """
        value = self.default
        if self.inherit and base_classes_meta is not None:
            value = getattr(base_classes_meta, self.name, value)
        if Meta is not None:
            value = getattr(Meta, self.name, value)
        return value