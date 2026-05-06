def netconf_state_sessions_session_session_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        netconf_state = ET.SubElement(config, "netconf-state", xmlns="urn:ietf:params:xml:ns:yang:ietf-netconf-monitoring")
        sessions = ET.SubElement(netconf_state, "sessions")
        session = ET.SubElement(sessions, "session")
        session_id = ET.SubElement(session, "session-id")
        session_id.text = kwargs.pop('session_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)