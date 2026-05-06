def diag_post_enable(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        diag = ET.SubElement(config, "diag", xmlns="urn:brocade.com:mgmt:brocade-diagnostics")
        post = ET.SubElement(diag, "post")
        enable = ET.SubElement(post, "enable")

        callback = kwargs.pop('callback', self._callback)
        return callback(config)