def extract_new(cls) -> DevicesTypeUnbound:
        """Gather all "new" |Node| or |Element| objects.

        See the main documentation on module |devicetools| for further
        information.
        """
        devices = cls.get_handlerclass()(*_selection[cls])
        _selection[cls].clear()
        return devices