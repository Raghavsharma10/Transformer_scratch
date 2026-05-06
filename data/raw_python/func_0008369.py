def from_protobuf(cls, proto: PluginInfoProto) -> PluginInfo:
        """
        Constructor from protobuf.

        :param proto: protobuf structure
        :type proto: ~unidown.plugin.protobuf.plugin_info_pb2.PluginInfoProto
        :return: the PluginInfo
        :rtype: ~unidown.plugin.plugin_info.PluginInfo
        :raises ValueError: name of PluginInfo does not exist or is empty inside the protobuf
        :raises ValueError: version of PluginInfo does not exist or is empty inside the protobuf
        :raises ValueError: host of PluginInfo does not exist or is empty inside the protobuf
        """
        if proto.name == "":
            raise ValueError("name of PluginInfo does not exist or is empty inside the protobuf.")
        elif proto.version == "":
            raise ValueError("version of PluginInfo does not exist or is empty inside the protobuf.")
        elif proto.host == "":
            raise ValueError("host of PluginInfo does not exist or is empty inside the protobuf.")
        return cls(proto.name, proto.version, proto.host)