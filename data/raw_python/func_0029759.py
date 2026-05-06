def terminal_cfg_line_exec_timeout(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        terminal_cfg = ET.SubElement(config, "terminal-cfg", xmlns="urn:brocade.com:mgmt:brocade-terminal")
        line = ET.SubElement(terminal_cfg, "line")
        sessionid_key = ET.SubElement(line, "sessionid")
        sessionid_key.text = kwargs.pop('sessionid')
        exec_timeout = ET.SubElement(line, "exec-timeout")
        exec_timeout.text = kwargs.pop('exec_timeout')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)