def _handle_node(self, node, scope=None, ctxt=None, stream=None):
        """Recursively handle nodes in the 010 AST

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO
        """
        if scope is None:
            if self._scope is None:
                self._scope = scope = self._create_scope()
            else:
                scope = self._scope

        if ctxt is None and self._ctxt is not None:
            ctxt = self._ctxt
        else:
            self._ctxt = ctxt

        if type(node) is tuple:
            node = node[1]

        # TODO probably a better way to do this...
        # this occurs with if-statements that have a single statement
        # instead of a compound statement (no curly braces)
        elif type(node) is list and len(list(filter(lambda x: isinstance(x, AST.Node), node))) == len(node):
            node = AST.Compound(
                block_items=node,
                coord=node[0].coord
            )
            return self._handle_node(node, scope, ctxt, stream)

        # need to check this so that debugger-eval'd statements
        # don't mess with the current state
        if not self._no_debug:
            self._coord = node.coord

        self._dlog("handling node type {}, line {}".format(node.__class__.__name__, node.coord.line if node.coord is not None else "?"))
        self._log.inc()

        breakable = self._node_is_breakable(node)

        if breakable and not self._no_debug and self._break_type != self.BREAK_NONE:
            # always break
            if self._break_type == self.BREAK_INTO:
                self._break_level = self._scope.level()
                self.debugger.cmdloop()

            # level <= _break_level
            elif self._break_type == self.BREAK_OVER:
                if self._scope.level() <= self._break_level:
                    self._break_level = self._scope.level()
                    self.debugger.cmdloop()
                else:
                    pass

        if node.__class__ not in self._node_switch:
            raise errors.UnsupportedASTNode(node.coord, node.__class__.__name__)

        res = self._node_switch[node.__class__](node, scope, ctxt, stream)

        self._log.dec()

        return res