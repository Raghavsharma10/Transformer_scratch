def delete_service(self, service_name, params=None):
        """
        Delete the service of the given name.  It may fail if there are
        any service keys or app bindings.  Use purge() if you want
        to delete it all.
        """
        if not self.space.has_service_with_name(service_name):
            logging.warning("Service not found so... succeeded?")
            return True

        guid = self.get_instance_guid(service_name)
        logging.info("Deleting service %s with guid %s" % (service_name, guid))

        # MAINT: this endpoint changes in newer version of api
        return self.api.delete("/v2/service_instances/%s?accepts_incomplete=true" %
            (guid), params=params)