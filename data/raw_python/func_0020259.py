def load_related(self, related, *related_fields):
        '''It returns a new :class:`Query` that automatically
follows the foreign-key relationship ``related``.

:parameter related: A field name corresponding to a :class:`ForeignKey`
    in :attr:`Query.model`.
:parameter related_fields: optional :class:`Field` names for the ``related``
    model to load. If not provided, all fields will be loaded.

This function is :ref:`performance boost <performance-loadrelated>` when
accessing the related fields of all (most) objects in your query.

If Your model contains more than one foreign key, you can use this function
in a generative way::

    qs = myquery.load_related('rel1').load_related('rel2','field1','field2')

:rtype: a new :class:`Query`.'''
        field = self._get_related_field(related)
        if not field:
            raise FieldError('"%s" is not a related field for "%s"' %
                             (related, self._meta))
        q = self._clone()
        return q._add_to_load_related(field, *related_fields)