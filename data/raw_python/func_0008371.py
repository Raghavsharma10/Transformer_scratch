def from_protobuf(cls, proto: SaveStateProto) -> SaveState:
        """
        Constructor from protobuf. Can raise ValueErrors from called from_protobuf() parsers.

        :param proto: protobuf structure
        :type proto: ~unidown.plugin.protobuf.save_state_pb2.SaveStateProto
        :return: the SaveState
        :rtype: ~unidown.plugin.save_state.SaveState
        :raises ValueError: version of SaveState does not exist or is empty inside the protobuf
        :raises ~packaging.version.InvalidVersion: version is not PEP440 conform
        """
        data_dict = {}
        for key, link_item in proto.data.items():
            data_dict[key] = LinkItem.from_protobuf(link_item)
        if proto.version == "":
            raise ValueError("version of SaveState does not exist or is empty inside the protobuf.")
        try:
            version = Version(proto.version)
        except InvalidVersion:
            raise InvalidVersion(f"Plugin version is not PEP440 conform: {proto.version}")
        return cls(version, PluginInfo.from_protobuf(proto.plugin_info), Timestamp.ToDatetime(proto.last_update),
                   data_dict)