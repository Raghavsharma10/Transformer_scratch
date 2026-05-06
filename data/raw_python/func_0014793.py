def create_from_data_channel(cls, data_channel):
        """Scan the data tree on the given data_channel to create a corresponding
        InputSetGenerator tree.
        """
        gather_depth = cls._get_gather_depth(data_channel)

        generator = InputSetGeneratorNode()
        for (data_path, data_node) in data_channel.get_ready_data_nodes(
                [], gather_depth):
            flat_data_node = data_node.flattened_clone(save=False)
            input_item = InputItem(
                flat_data_node, data_channel.channel,
                data_channel.as_channel, mode=data_channel.mode)
            generator._add_input_item(data_path, input_item)
        return generator