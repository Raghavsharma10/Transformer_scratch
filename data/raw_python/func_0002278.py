def parse(cls, parser, token):
        """
        Parse the node syntax:

        .. code-block:: html+django

            {% page_placeholder parentobj slotname title="test" role="m" %}
        """
        bits, as_var = parse_as_var(parser, token)
        tag_name, args, kwargs = parse_token_kwargs(parser, bits, allowed_kwargs=cls.allowed_kwargs, compile_args=True, compile_kwargs=True)

        # Play with the arguments
        if len(args) == 2:
            parent_expr = args[0]
            slot_expr = args[1]
        elif len(args) == 1:
            # Allow 'page' by default. Works with most CMS'es, including django-fluent-pages.
            parent_expr = Variable('page')
            slot_expr = args[0]
        else:
            raise TemplateSyntaxError("""{0} tag allows two arguments: 'parent object' 'slot name' and optionally: title=".." role="..".""".format(tag_name))

        cls.validate_args(tag_name, *args, **kwargs)
        return cls(
            tag_name=tag_name,
            as_var=as_var,
            parent_expr=parent_expr,
            slot_expr=slot_expr,
            **kwargs
        )