def get_prep_lookup(self, lookup_name, rhs):
        """
        Perform preliminary non-db specific lookup checks and conversions
        """
        if lookup_name == 'exact':
            if not isinstance(rhs, Model):
                raise FilteredGenericForeignKeyFilteringException(
                    "For exact lookup, please pass a single Model instance.")

        elif lookup_name in ['in', 'in_raw']:
            if type(rhs) == QuerySet:
                return rhs, None

            if not is_iterable(rhs):
                raise FilteredGenericForeignKeyFilteringException(
                    "For 'in' lookup, please pass an iterable or a QuerySet.")

        else:
            raise FilteredGenericForeignKeyFilteringException(
                "Lookup %s not supported." % lookup_name)

        return rhs, None