def _init_record(self, record_type_idstr):
        """Initializes a query record"""
        record_type_data = self._all_supported_record_type_data_sets[Id(record_type_idstr).get_identifier()]
        module = importlib.import_module(record_type_data['module_path'])
        record = getattr(module, record_type_data['query_record_class_name'])
        self._records[record_type_idstr] = record(self)