def aggregate(self, kwargs):
        '''Aggregate lookup parameters.'''
        meta = self._meta
        fields = meta.dfields
        field_lookups = {}
        for name, value in iteritems(kwargs):
            bits = name.split(JSPLITTER)
            field_name = bits.pop(0)
            if field_name not in fields:
                raise QuerySetError('Could not filter on model "{0}".\
 Field "{1}" does not exist.'.format(meta, field_name))
            field = fields[field_name]
            attname = field.attname
            lookup = None
            if bits:
                bits = [n.lower() for n in bits]
                if bits[-1] == 'in':
                    bits.pop()
                elif bits[-1] in range_lookups:
                    lookup = bits.pop()
                remaining = JSPLITTER.join(bits)
                if lookup:  # this is a range lookup
                    attname, nested = field.get_lookup(remaining,
                                                       QuerySetError)
                    lookups = get_lookups(attname, field_lookups)
                    lookups.append(lookup_value(lookup, (value, nested)))
                    continue
                elif remaining:   # Not a range lookup, must be a nested filter
                    value = field.filter(self.session, remaining, value)
            lookups = get_lookups(attname, field_lookups)
            # If we are here the field must be an index
            if not field.index:
                raise QuerySetError("%s %s is not an index. Cannot query." %
                                    (field.__class__.__name__, field_name))
            if not iterable(value):
                value = (value,)
            for v in value:
                if isinstance(v, Q):
                    v = lookup_value('set', v.construct())
                else:
                    v = lookup_value('value', field.serialise(v, lookup))
                lookups.append(v)
        #
        return [queryset(self, name=name, underlying=field_lookups[name])
                for name in sorted(field_lookups)]