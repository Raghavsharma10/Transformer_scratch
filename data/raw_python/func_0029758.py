def terminal_cfg_line_sessionid(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        terminal_cfg = ET.SubElement(config, "terminal-cfg", xmlns="urn:brocade.com:mgmt:brocade-terminal")
        line = ET.SubElement(terminal_cfg, "line")
        sessionid = ET.SubElement(line, "sessionid")
        sessionid.text = kwargs.pop('sessionid')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)