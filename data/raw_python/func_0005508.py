def set_deployment_id(self):
        """Sets the deployment ID from deployment properties

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_deployment_id')
        deployment_id_val = self.get_value('cons3rt.deployment.id')
        if not deployment_id_val:
            log.debug('Deployment ID not found in deployment properties')
            return
        try:
            deployment_id = int(deployment_id_val)
        except ValueError:
            log.debug('Deployment ID found was unable to convert to an int: {d}'.format(d=deployment_id_val))
            return
        self.deployment_id = deployment_id
        log.info('Found deployment ID: {i}'.format(i=str(self.deployment_id)))