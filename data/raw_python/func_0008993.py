def select_storage_for(cls, section_name, storage):
        """Selects the data storage for a config section within the
        :param:`storage`. The primary config section is normally merged into
        the :param:`storage`.

        :param section_name:    Config section (name) to process.
        :param storage:         Data storage to use.
        :return: :param:`storage` or a part of it (as section storage).
        """
        section_storage = storage
        storage_name = cls.get_storage_name_for(section_name)
        if storage_name:
            section_storage = storage.get(storage_name, None)
            if section_storage is None:
                section_storage = storage[storage_name] = dict()
        return section_storage