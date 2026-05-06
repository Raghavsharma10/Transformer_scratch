def log(self, content, content_type):
        """Logs an item.

        This method is a shortcut to ``createLogEntry()``.

        arg:    content (object): the entry to log
        arg:    content_type (osid.type.Type): the type of this entry
                which must be one of the types returned by
                ``LoggingManager.getContentTypes()``
        raise:  InvalidArgument - ``content`` is not of ``content_type``
        raise:  NullArgument - ``content`` or ``content_type`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported -
                ``LoggingManager.supportsContentType(contentType)`` is
                ``false``
        *compliance: mandatory -- This method must be implemented.*

        """
        if content_type not in self._content_types:
            raise errors.Unsupported()
        lefc = self._leas.get_content_form_for_create([])
        lefc.set_timestamp(DateTime.utcnow())