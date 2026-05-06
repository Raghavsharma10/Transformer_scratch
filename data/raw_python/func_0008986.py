def parse_config_section(config_section, section_schema):
    """Parse a config file section (INI file) by using its schema/description.

    .. sourcecode::

        import configparser     # -- NOTE: Use backport for Python2
        import click
        from click_configfile import SectionSchema, Param, parse_config_section

        class ConfigSectionSchema(object):
            class Foo(SectionSchema):
                name    = Param(type=str)
                flag    = Param(type=bool)
                numbers = Param(type=int, multiple=True)
                filenames = Param(type=click.Path(), multiple=True)

        parser = configparser.ConfigParser()
        parser.read(["foo.ini"])
        config_section = parser["foo"]
        data = parse_config_section(config_section, ConfigSectionSchema.Foo)
        # -- FAILS WITH: click.BadParameter if conversion errors occur.

    .. sourcecode:: ini

        # -- FILE: foo.ini
        [foo]
        name = Alice
        flag = yes      # true, false, yes, no (case-insensitive)
        numbers = 1 4 9 16 25
        filenames = foo/xxx.txt
            bar/baz/zzz.txt

    :param config_section:  Config section to parse
    :param section_schema:  Schema/description of config section (w/ Param).
    :return: Retrieved data, values converted to described types.
    :raises: click.BadParameter, if conversion error occurs.
    """
    storage = {}
    for name, param in select_params_from_section_schema(section_schema):
        value = config_section.get(name, None)
        if value is None:
            if param.default is None:
                continue
            value = param.default
        else:
            value = param.parse(value)
        # -- DIAGNOSTICS:
        # print("  %s = %s" % (name, repr(value)))
        storage[name] = value
    return storage