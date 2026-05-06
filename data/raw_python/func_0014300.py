def _extract_data(self, request):
        """Extract data from POST.

        Handles extracting a vanilla Python dict of values that are present
        in the given model. This also handles instances of ``ForeignKey`` and
        will convert those to the appropriate object instances from the
        database. In other words, it will see that user is a ``ForeignKey`` to
        Django's ``User`` class, assume the value is an appropriate pk, and
        load up that record.
        """
        data = {}
        for field, val in six.iteritems(request.POST):
            if field in self.immutable_fields:
                continue  # Ignore immutable fields silently.

            if field in self.fields:
                field_obj = self.model._meta.get_field(field)
                val = self._extract_value(val)
                if isinstance(field_obj, models.ForeignKey):
                    if field_obj.null and not val:
                        clean_value = None
                    else:
                        clean_value = field_obj.rel.to.objects.get(pk=val)
                else:
                    clean_value = field_obj.to_python(val)
                data[smart_str(field)] = clean_value

        return data