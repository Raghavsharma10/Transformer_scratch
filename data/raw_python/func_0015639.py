def add_objects_from_string(self, buffer, object_ids):
        """add_objects_from_string(buffer, object_ids)

        :param buffer: the string to parse
        :type buffer: :obj:`str`
        :param object_ids: array of objects to build
        :type object_ids: [:obj:`str`]

        :raises: :class:`GLib.Error`

        :returns: A positive value on success, 0 if an error occurred
        :rtype: :obj:`int`

        {{ docs }}
        """

        length = -1
        return Gtk.Builder.add_objects_from_string(self, buffer, length, object_ids)