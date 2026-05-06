def set_virtualization_realm_type(self):
        """Sets the virtualization realm type from deployment properties

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_virtualization_realm_type')
        self.virtualization_realm_type = self.get_value('cons3rt.deploymentRun.virtRealm.type')
        log.info('Found virtualization realm type : {t}'.format(t=self.virtualization_realm_type))