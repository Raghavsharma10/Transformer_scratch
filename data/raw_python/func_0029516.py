def netconf_state_datastores_datastore_transaction_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        datastores = ET.SubElement(netconf_state, "datastores")
        datastore = ET.SubElement(datastores, "datastore")
        name_key = ET.SubElement(datastore, "name")
        name_key.text = kwargs.pop('name')
        transaction_id = ET.SubElement(datastore, "transaction-id", xmlns="http://tail-f.com/yang/netconf-monitoring")
        transaction_id.text = kwargs.pop('transaction_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)