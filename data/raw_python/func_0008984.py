def assign_param_names(cls=None, param_class=None):
    """Class decorator to assign parameter name to instances of :class:`Param`.

    .. sourcecode::

        @assign_param_names
        class ConfigSectionSchema(object):
            alice = Param(type=str)
            bob   = Param(type=str)

        assert ConfigSectionSchema.alice.name == "alice"
        assert ConfigSectionSchema.bob.name == "bob"

    .. sourcecode::

        # -- NESTED ASSIGN: Covers also nested SectionSchema subclasses.
        @assign_param_names
        class ConfigSectionSchema(object):
            class Foo(SectionSchema):
                alice = Param(type=str)
                bob   = Param(type=str)

        assert ConfigSectionSchema.Foo.alice.name == "alice"
        assert ConfigSectionSchema.Foo.bob.name == "bob"
    """
    if param_class is None:
        param_class = Param

    def decorate_class(cls):
        for name, value in select_params_from_section_schema(cls, param_class,
                                                             deep=True):
            # -- ANNOTATE PARAM: By assigning its name
            if not value.name:
                value.name = name
        return cls

    # -- DECORATOR LOGIC:
    if cls is None:
        # -- CASE: @assign_param_names
        # -- CASE: @assign_param_names(...)
        return decorate_class
    else:
        # -- CASE: @assign_param_names class X: ...
        # -- CASE: assign_param_names(my_class)
        # -- CASE: my_class = assign_param_names(my_class)
        return decorate_class(cls)