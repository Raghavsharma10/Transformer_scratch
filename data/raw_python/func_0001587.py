def register(conf, conf_admin, **options):
    """
    Register a new admin section.

    :param conf: A subclass of ``djconfig.admin.Config``
    :param conf_admin: A subclass of ``djconfig.admin.ConfigAdmin``
    :param options: Extra options passed to ``django.contrib.admin.site.register``
    """
    assert issubclass(conf_admin, ConfigAdmin), (
        'conf_admin is not a ConfigAdmin subclass')
    assert issubclass(
        getattr(conf_admin, 'change_list_form', None),
        ConfigForm), 'No change_list_form set'
    assert issubclass(conf, Config), (
        'conf is not a Config subclass')
    assert conf.app_label, 'No app_label set'
    assert conf.verbose_name_plural, 'No verbose_name_plural set'
    assert not conf.name or re.match(r"^[a-zA-Z_]+$", conf.name), (
        'Not a valid name. Valid chars are [a-zA-Z_]')
    config_class = type("Config", (), {})
    config_class._meta = type("Meta", (_ConfigMeta,), {
        'app_label': conf.app_label,
        'verbose_name_plural': conf.verbose_name_plural,
        'object_name': 'Config',
        'model_name': conf.name,
        'module_name': conf.name})
    admin.site.register([config_class], conf_admin, **options)