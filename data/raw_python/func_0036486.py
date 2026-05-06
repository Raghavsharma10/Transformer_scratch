def get_markup_choices():
    """
    Receives available markup options as list.
    """
    available_reader_list = []
    module_dir = os.path.realpath(os.path.dirname(__file__))
    module_names = filter(
        lambda x: x.endswith('_reader.py'), os.listdir(module_dir))

    for module_name in module_names:
        markup = module_name.split('_')[0]
        reader = get_reader(markup=markup)

        if reader.enabled is True:
            available_reader_list.append((markup, reader.name))

    return available_reader_list