def _handle_watch_metadata(self, node, scope, ctxt, stream):
        """Handle watch vars for fields
        """
        keyvals = node.metadata.keyvals
        if "watch" not in keyvals:
            raise errors.PfpError("Packed fields require a packer function set")
        if "update" not in keyvals:
            raise errors.PfpError("Packed fields require a packer function set")

        watch_field_name = keyvals["watch"]
        update_func_name = keyvals["update"]

        watch_fields = list(map(lambda x: self.eval(x.strip()), watch_field_name.split(";")))
        update_func = scope.get_id(update_func_name)

        return {
            "type": "watch",
            "watch_fields": watch_fields,
            "update_func": update_func,
            "func_call_info": (ctxt, scope, stream, self, self._coord)
        }