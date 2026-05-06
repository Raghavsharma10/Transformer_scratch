def _handle_struct_ref(self, node, scope, ctxt, stream):
        """TODO: Docstring for _handle_struct_ref.

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling struct ref")

        # name
        # field
        struct = self._handle_node(node.name, scope, ctxt, stream)

        try:
            sub_field = getattr(struct, node.field.name)
        except AttributeError as e:
            # should be able to access implicit array items by index OR
            # access the last one's members directly without index
            #
            # E.g.:
            # 
            # local int total_length = 0;
            # while(!FEof()) {
            #     HEADER header;
            #   total_length += header.length;
            # }
            if isinstance(struct, fields.Array) and struct.implicit:
                last_item = struct[-1]
                sub_field = getattr(last_item, node.field.name)
            else:
                raise

        return sub_field