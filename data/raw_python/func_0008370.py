def to_protobuf(self) -> PluginInfoProto:
        """
        Create protobuf item.

        :return: protobuf structure
        :rtype: ~unidown.plugin.protobuf.link_item_pb2.PluginInfoProto
        """
        proto = PluginInfoProto()
        proto.name = self.name
        proto.version = str(self.version)
        proto.host = self.host
        return proto