def set_deployment_name(self):
        """Sets the deployment name from deployment properties

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_deployment_name')
        self.deployment_name = self.get_value('cons3rt.deployment.name')
        log.info('Found deployment name: {n}'.format(n=self.deployment_name))