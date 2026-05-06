def _init_record(self, record_type_idstr):
        """Override this from osid.Extensible because Forms use a different
        attribute in record_type_data."""
        record_type_data = self._record_type_data_sets[Id(record_type_idstr).get_identifier()]
        module = importlib.import_module(record_type_data['module_path'])
        record = getattr(module, record_type_data['form_record_class_name'])
        if record is not None:
            self._records[record_type_idstr] = record(self)
            return True
        else:
            return False