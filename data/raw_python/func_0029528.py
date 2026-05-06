def netconf_state_sessions_session_out_rpc_errors(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        sessions = ET.SubElement(netconf_state, "sessions")
        session = ET.SubElement(sessions, "session")
        session_id_key = ET.SubElement(session, "session-id")
        session_id_key.text = kwargs.pop('session_id')
        out_rpc_errors = ET.SubElement(session, "out-rpc-errors")
        out_rpc_errors.text = kwargs.pop('out_rpc_errors')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)