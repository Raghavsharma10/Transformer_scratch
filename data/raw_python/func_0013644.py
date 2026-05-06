def textfsm_extractor(cls, template_name, raw_text):
    """
    Applies a TextFSM template over a raw text and return the matching table.

    Main usage of this method will be to extract data form a non-structured output
    from a network device and return the values in a table format.

    :param cls: Instance of the driver class
    :param template_name: Specifies the name of the template to be used
    :param raw_text: Text output as the devices prompts on the CLI
    :return: table-like list of entries
    """
    textfsm_data = list()
    cls.__class__.__name__.replace('Driver', '')
    current_dir = os.path.dirname(os.path.abspath(sys.modules[cls.__module__].__file__))
    template_dir_path = '{current_dir}/utils/textfsm_templates'.format(
        current_dir=current_dir
    )
    template_path = '{template_dir_path}/{template_name}.tpl'.format(
        template_dir_path=template_dir_path,
        template_name=template_name
    )

    try:
        fsm_handler = textfsm.TextFSM(open(template_path))
    except IOError:
        raise napalm_base.exceptions.TemplateNotImplemented(
            "TextFSM template {template_name}.tpl is not defined under {path}".format(
                template_name=template_name,
                path=template_dir_path
            )
        )
    except textfsm.TextFSMTemplateError as tfte:
        raise napalm_base.exceptions.TemplateRenderException(
            "Wrong format of TextFSM template {template_name}: {error}".format(
                template_name=template_name,
                error=py23_compat.text_type(tfte)
            )
        )

    objects = fsm_handler.ParseText(raw_text)

    for obj in objects:
        index = 0
        entry = {}
        for entry_value in obj:
            entry[fsm_handler.header[index].lower()] = entry_value
            index += 1
        textfsm_data.append(entry)

    return textfsm_data