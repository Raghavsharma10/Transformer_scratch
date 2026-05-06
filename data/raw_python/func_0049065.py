def _get_record(self, record_type):
        """This overrides _get_record in osid.Extensible.

        Perhaps we should leverage it somehow?

        """
        if (not self.has_record_type(record_type) and
                record_type.get_identifier() not in self._record_type_data_sets):
            raise errors.Unsupported()
        if str(record_type) not in self._records:
            record_initialized = self._init_record(str(record_type))
            if record_initialized and str(record_type) not in self._my_map['recordTypeIds']:
                self._my_map['recordTypeIds'].append(str(record_type))
        return self._records[str(record_type)]