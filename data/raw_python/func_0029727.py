def get_port_channel_detail_input_last_aggregator_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        get_port_channel_detail = ET.Element("get_port_channel_detail")
        config = get_port_channel_detail
        input = ET.SubElement(get_port_channel_detail, "input")
        last_aggregator_id = ET.SubElement(input, "last-aggregator-id")
        last_aggregator_id.text = kwargs.pop('last_aggregator_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)