def logging_auditlog_class_class(self, **kwargs):
        """Auto Generated Code
        """
        config = ET.Element("config")
        logging = ET.SubElement(config, "logging", xmlns="urn:brocade.com:mgmt:brocade-ras")
        auditlog = ET.SubElement(logging, "auditlog")
        class_el = ET.SubElement(auditlog, "class")
        class_el = ET.SubElement(class_el, "class")
        class_el.text = kwargs.pop('class')

        callback = kwargs.pop('callback', self._callback)
        return callback(config)