def _handle_packed_metadata(self, node, scope, ctxt, stream):
        """Handle packed metadata
        """
        keyvals = node.metadata.keyvals
        if "packer" not in keyvals and ("pack" not in keyvals or "unpack" not in keyvals):
            raise errors.PfpError("Packed fields require a packer function to be set or pack and unpack functions to be set")
        if "packtype" not in keyvals:
            raise errors.PfpError("Packed fields require a packtype to be set")

        args_ = {}
        if "packer" in keyvals:
            packer_func_name = keyvals["packer"]
            packer_func = scope.get_id(packer_func_name)
            args_["packer"] = packer_func
        elif "pack" in keyvals and "unpack" in keyvals:
            pack_func = scope.get_id(keyvals["pack"])
            unpack_func = scope.get_id(keyvals["unpack"])
            args_["pack"] = pack_func
            args_["unpack"] = unpack_func

        packtype_cls_name = keyvals["packtype"]
        packtype_cls = scope.get_type(packtype_cls_name)
        args_["pack_type"] = packtype_cls

        args_["type"] = "packed"
        args_["func_call_info"] = (ctxt, scope, stream, self, self._coord)
        return args_