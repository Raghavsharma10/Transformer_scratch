def process_factory_meta_options(
        mcs_args: McsArgs,
        default_factory_class: Type[MetaOptionsFactory] = MetaOptionsFactory,
        factory_attr_name: str = META_OPTIONS_FACTORY_CLASS_ATTR_NAME) \
        -> MetaOptionsFactory:
    """
    Main entry point for consumer metaclasses. Usage::

        from py_meta_utils import (AbstractMetaOption, McsArgs, MetaOptionsFactory,
                                   process_factory_meta_options)


        class YourMetaOptionsFactory(MetaOptionsFactory):
            _options = [AbstractMetaOption]


        class YourMetaclass(type):
            def __new__(mcs, name, bases, clsdict):
                mcs_args = McsArgs(mcs, name, bases, clsdict)

                # process_factory_meta_options must come *before* super().__new__()
                process_factory_meta_options(mcs_args, YourMetaOptionsFactory)
                return super().__new__(*mcs_args)


        class YourClass(metaclass=YourMetaclass):
            pass

    Subclasses of ``YourClass`` may set their ``_meta_options_factory_class``
    attribute to a subclass of ``YourMetaOptionsFactory`` to customize
    their own supported meta options::

        from py_meta_utils import MetaOption


        class FooMetaOption(MetaOption):
            def __init__(self):
                super().__init__(name='foo', default=None, inherit=True)


        class FooMetaOptionsFactory(YourMetaOptionsFactory):
            _options = YourMetaOptionsFactory._options + [
                FooMetaOption,
            ]


        class FooClass(YourClass):
            _meta_options_factory_class = FooMetaOptionsFactory

            class Meta:
                foo = 'bar'

    :param mcs_args: The :class:`McsArgs` for the class-under-construction
    :param default_factory_class: The default MetaOptionsFactory class to use, if
                                  the ``factory_attr_name`` attribute is not set on
                                  the class-under-construction
    :param factory_attr_name: The attribute name to look for an overridden factory
                              meta options class on the class-under-construction
    :return: The populated instance of the factory class
    """
    factory_cls = mcs_args.getattr(
        factory_attr_name or META_OPTIONS_FACTORY_CLASS_ATTR_NAME,
        default_factory_class)
    options_factory = factory_cls()
    options_factory._contribute_to_class(mcs_args)
    return options_factory