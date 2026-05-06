def to_protobuf(self) -> SaveStateProto:
        """
        Create protobuf item.

        :return: protobuf structure
        :rtype: ~unidown.plugin.protobuf.save_state_pb2.SaveStateProto
        """
        result = SaveStateProto()
        result.version = str(self.version)
        result.last_update.CopyFrom(datetime_to_timestamp(self.last_update))
        result.plugin_info.CopyFrom(self.plugin_info.to_protobuf())
        for key, link_item in self.link_item_dict.items():
            result.data[key].CopyFrom(link_item.to_protobuf())
        return result