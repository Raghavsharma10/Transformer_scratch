def from_protobuf(cls, proto: LinkItemProto) -> LinkItem:
        """
        Constructor from protobuf.

        :param proto: protobuf structure
        :type proto: ~unidown.plugin.protobuf.link_item_pb2.LinkItemProto
        :return: the LinkItem
        :rtype: ~unidown.plugin.link_item.LinkItem
        :raises ValueError: name of LinkItem does not exist inside the protobuf or is empty
        """
        if proto.name == '':
            raise ValueError("name of LinkItem does not exist or is empty inside the protobuf.")
        return cls(proto.name, Timestamp.ToDatetime(proto.time))