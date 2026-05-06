def get_assessment_part_lookup_session(runtime, proxy, section=None):
    """returns an assessment part lookup session, perhaps even a magic one"""
    # This appears to share code with get_item_lookup_session
    try:
        config = runtime.get_configuration()
        parameter_id = Id('parameter:magicAssessmentPartLookupSessions@json')
        import_path_with_class = config.get_value_by_parameter(parameter_id).get_string_value()
        module_path = '.'.join(import_path_with_class.split('.')[0:-1])
        magic_class = import_path_with_class.split('.')[-1]
        module = importlib.import_module(module_path)
        part_lookup_session = getattr(module, magic_class)(section,
                                                           runtime=runtime,
                                                           proxy=proxy)
    except (AttributeError, KeyError, NotFound):
        mgr = get_provider_manager('ASSESSMENT_AUTHORING',
                                   runtime=runtime,
                                   proxy=proxy,
                                   local=True)
        part_lookup_session = mgr.get_assessment_part_lookup_session(proxy=proxy)
    return part_lookup_session