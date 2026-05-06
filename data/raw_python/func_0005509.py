def set_deployment_run_name(self):
        """Sets the deployment run name from deployment properties

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_deployment_run_name')
        self.deployment_run_name = self.get_value('cons3rt.deploymentRun.name')
        log.info('Found deployment run name: {n}'.format(n=self.deployment_run_name))