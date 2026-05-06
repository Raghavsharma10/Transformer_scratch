def delete_service_key(self, service_name, key_name):
        """
        Delete a service key for the given service.
        """
        key = self.get_service_key(service_name, key_name)
        logging.info("Deleting service key %s for service %s" % (key, service_name))
        return self.api.delete(key['metadata']['url'])