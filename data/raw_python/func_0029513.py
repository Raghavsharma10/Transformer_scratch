def netconf_state_datastores_datastore_locks_lock_type_global_lock_global_lock_locked_time(self, **kwargs):
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
        global_lock = ET.SubElement(lock_type, "global-lock")
        global_lock = ET.SubElement(global_lock, "global-lock")
        locked_time = ET.SubElement(global_lock, "locked-time")
        locked_time.text = kwargs.pop('locked_time')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)