def get_registry(entry, runtime):
    """Returns a record registry given an entry and runtime"""
    try:
        records_location_param_id = Id('parameter:recordsRegistry@mongo')
        registry = runtime.get_configuration().get_value_by_parameter(
            records_location_param_id).get_string_value()
        return import_module(registry).__dict__.get(entry, {})
    except (ImportError, AttributeError, KeyError, NotFound):
        return {}