def matches_section(section_name):
    """Decorator for SectionSchema classes to define the mapping between
    a config section schema class and one or more config sections with
    matching name(s).

    .. sourcecode::

        @matches_section("foo")
        class FooSchema(SectionSchema):
            pass

        @matches_section(["bar", "baz.*"])
        class BarAndBazSchema(SectionSchema):
            pass

    .. sourcecode:: ini

        # -- FILE: *.ini
        [foo]       # USE: FooSchema
        ...

        [bar]       # USE: BarAndBazSchema
        ...

        [baz.alice] # USE: BarAndBazSchema
        ...
    """
    section_names = section_name
    if isinstance(section_name, six.string_types):
        section_names = [section_name]
    elif not isinstance(section_name, (list, tuple)):
        raise ValueError("%r (expected: string, strings)" % section_name)

    def decorator(cls):
        class_section_names = getattr(cls, "section_names", None)
        if class_section_names is None:
            cls.section_names = list(section_names)
        else:
            # -- BETTER SUPPORT: For multiple decorators
            #   @matches_section("foo")
            #   @matches_section("bar.*")
            #   class Example(SectionSchema):
            #       pass
            #   assert Example.section_names == ["foo", "bar.*"]
            approved = [name for name in section_names
                        if name not in cls.section_names]
            cls.section_names = approved + cls.section_names
        return cls
    return decorator