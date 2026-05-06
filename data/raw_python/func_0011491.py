def _handle_typedef(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_typedef.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        is_union_or_struct = (node.type.type.__class__ in [AST.Union, AST.Struct])
        is_enum = (node.type.type.__class__ is AST.Enum)

        if is_union_or_struct:
            self._dlog("handling typedef struct/union '{}'".format(node.name))
            scope.add_type_struct_or_union(node.name, self, node.type.type)
        elif is_enum:
            enum_cls = self._handle_node(node.type, scope, ctxt, stream)
            scope.add_type_class(node.name, enum_cls)
        elif isinstance(node.type, AST.ArrayDecl):
            # this does not parse data, just creates the ArrayDecl class
            array_cls = self._handle_node(node.type, scope, ctxt, stream)
            scope.add_type_class(node.name, array_cls)
        else:
            names = node.type.type.names

            self._dlog("handling typedef '{}' ({})".format(node.name, names))
            # don't actually handle the TypeDecl and Identifier nodes,
            # just directly add the types. Example structure:
            #
            #     Typedef: BLAH, [], ['typedef']
            #        TypeDecl: BLAH, []
            #            IdentifierType: ['unsigned', 'char']
            #    
            scope.add_type(node.name, names)