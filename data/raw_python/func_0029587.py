def diag_post_rbridge_id_rbridge_id(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        diag = ET.SubElement(config, "diag", xmlns="urn:brocade.com:mgmt:brocade-diagnostics")
        post = ET.SubElement(diag, "post")
        rbridge_id = ET.SubElement(post, "rbridge-id")
        rbridge_id = ET.SubElement(rbridge_id, "rbridge-id")
        rbridge_id.text = kwargs.pop('rbridge_id')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)