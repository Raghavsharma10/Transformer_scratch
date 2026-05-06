def to_protobuf(self) -> LinkItemProto:
        """
        Create protobuf item.

        :return: protobuf structure
        :rtype: ~unidown.plugin.protobuf.link_item_pb2.LinkItemProto
        """
        result = LinkItemProto()
        result.name = self._name
        result.time.CopyFrom(datetime_to_timestamp(self._time))
        return result