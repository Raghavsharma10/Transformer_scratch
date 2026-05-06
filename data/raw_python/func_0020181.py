def get_lookup(self, remaining, errorClass=ValueError):
        '''called by the :class:`Query` method when it needs to build
lookup on fields with additional nested fields. This is the case of
:class:`ForeignKey` and :class:`JSONField`.

:param remaining: the :ref:`double underscored` fields if this :class:`Field`
:param errorClass: Optional exception class to use if the *remaining* field
    is not valid.'''
        if remaining:
            raise errorClass('Cannot use nested lookup on field %s' % self)
        return (self.attname, None)