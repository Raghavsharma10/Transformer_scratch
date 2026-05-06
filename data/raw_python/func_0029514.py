def netconf_state_datastores_datastore_locks_lock_type_partial_lock_partial_lock_lock_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        datastores = ET.SubElement(netconf_state, "datastores")
        datastore = ET.SubElement(datastores, "datastore")
        name_key = ET.SubElement(datastore, "name")
        name_key.text = kwargs.pop('name')
        locks = ET.SubElement(datastore, "locks")
        lock_type = ET.SubElement(locks, "lock-type")
        partial_lock = ET.SubElement(lock_type, "partial-lock")
        partial_lock = ET.SubElement(partial_lock, "partial-lock")
        lock_id = ET.SubElement(partial_lock, "lock-id")
        lock_id.text = kwargs.pop('lock_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)