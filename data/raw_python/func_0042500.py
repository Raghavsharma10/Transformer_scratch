def coalesce_outputs(tree):
    """
    Coalesce the constant output expressions

        __output__('foo')
        __output__('bar')
        __output__(baz)
        __output__('xyzzy')

    into

        __output__('foobar', baz, 'xyzzy')
    """

    coalesce_all_outputs = True
    if coalesce_all_outputs:
        should_coalesce = lambda n: True
    else:
        should_coalesce = lambda n: n.output_args[0].__class__ is Str

    class OutputCoalescer(NodeVisitor):
        def visit(self, node):
            # if - else expression also has a body! it is not we want, though.
            if hasattr(node, 'body') and isinstance(node.body, Iterable):
                # coalesce continuous string output nodes
                new_body = []
                output_node = None

                def coalesce_strs():
                    if output_node:
                        output_node.value.args[:] = \
                            coalesce_strings(output_node.value.args)

                for i in node.body:
                    if hasattr(i, 'output_args') and should_coalesce(i):
                        if output_node:
                            if len(output_node.value.args) + len(i.output_args) > 250:
                                coalesce_strs()
                                output_node = i
                            else:
                                output_node.value.args.extend(i.output_args)
                                continue

                        output_node = i

                    else:
                        coalesce_strs()
                        output_node = None

                    new_body.append(i)

                coalesce_strs()
                node.body[:] = new_body

            NodeVisitor.visit(self, node)

        def check(self, node):
            """
            Coalesce __TK__output(__TK__escape(literal(x))) into
            __TK__output(x).
            """
            if not ast_equals(node.func, NameX('__TK__output')):
                return

            for i in range(len(node.args)):
                arg1 = node.args[i]
                if not arg1.__class__.__name__ == 'Call':
                    continue

                if not ast_equals(arg1.func, NameX('__TK__escape')):
                    continue

                if len(arg1.args) != 1:
                    continue

                arg2 = arg1.args[0]
                if not arg2.__class__.__name__ == 'Call':
                    continue

                if not ast_equals(arg2.func, NameX('literal')):
                    continue

                if len(arg2.args) != 1:
                    continue

                node.args[i] = arg2.args[0]

        def visit_Call(self, node):
            self.check(node)
            self.generic_visit(node)

    OutputCoalescer().visit(tree)