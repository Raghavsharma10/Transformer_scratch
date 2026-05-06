def _delete_service(self, service_only=False):
        """
        Delete a Cloud Foundry service and any associations.
        """
        logging.debug('_delete_service()')
        return self.service.delete_service(self.service_name)