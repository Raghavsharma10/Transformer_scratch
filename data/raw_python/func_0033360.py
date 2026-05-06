def fieldmap(self):
        '''
        Dictionary of field_id: field_name, as defined in self.fields property
        '''
        if hasattr(self, '_sql_fieldmap') and self._sql_fieldmap:
            fieldmap = self._sql_fieldmap
        else:
            fieldmap = defaultdict(str)
            fields = copy(self.fields)
            for field, opts in fields.iteritems():
                field_id = opts.get('what')
                if field_id is not None:
                    fieldmap[field_id] = field
            self._sql_fieldmap = fieldmap
        return fieldmap