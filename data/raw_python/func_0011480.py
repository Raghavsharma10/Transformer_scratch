def _handle_metadata(self, node, scope, ctxt, stream):
        """Handle metadata for the node
        """
        self._dlog("handling node metadata {}".format(node.metadata.keyvals))

        keyvals = node.metadata.keyvals

        metadata_info = []

        if "watch" in node.metadata.keyvals or "update" in keyvals:
            metadata_info.append(
                self._handle_watch_metadata(node, scope, ctxt, stream)
            )

        if "packtype" in node.metadata.keyvals or "packer" in keyvals:
            metadata_info.append(
                self._handle_packed_metadata(node, scope, ctxt, stream)
            )

        return metadata_info