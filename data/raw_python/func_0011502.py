def _handle_param_list(self, node, scope, ctxt, stream):
        """Handle ParamList nodes

        :node: TODO
        :scope: TODO
        :ctxt: TODO
        :stream: TODO
        :returns: TODO

        """
        self._dlog("handling param list")
        # params should be a list of tuples:
        # [(<name>, <field_class>), ...]
        params = []
        for param in node.params:
            self._mark_id_as_lazy(param)
            param_info = self._handle_node(param, scope, ctxt, stream)
            params.append(param_info)

        param_list = functions.ParamListDef(params, node.coord)
        return param_list