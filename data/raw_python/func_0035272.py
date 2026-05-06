def walk(self, dispatcher, node):
        """
        Walk through the node with a custom dispatcher for extraction of
        details that are required.
        """

        deferrable_handlers = {
            Declare: self.declare,
            Resolve: self.register_reference,
        }
        layout_handlers = {
            PushScope: self.push_scope,
            PopScope: self.pop_scope,
            PushCatch: self.push_catch,
            # should really be different, but given that the
            # mechanism is within the same tree, the only difference
            # would be sanity check which should have been tested in
            # the first place in the primitives anyway.
            PopCatch: self.pop_scope,
        }

        if not self.shadow_funcname:
            layout_handlers[ResolveFuncName] = self.shadow_reference

        local_dispatcher = Dispatcher(
            definitions=dict(dispatcher),
            token_handler=None,
            layout_handlers=layout_handlers,
            deferrable_handlers=deferrable_handlers,
        )
        return list(walk(local_dispatcher, node))