def get_or_create(self, event):
        """Gets or creates a new event hook for the specified event (key).

        This method treats qcore.EnumBase-typed event keys specially:
        enum_member.name is used as key instead of enum instance
        in case such a key is passed.

        Note that on/off/trigger/safe_trigger methods rely on this method,
        so you can pass enum members there as well.

        """
        if isinstance(event, EnumBase):
            event = event.short_name
        return self.__dict__.setdefault(event, EventHook())