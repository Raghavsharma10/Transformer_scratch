def purge(self):
        """
        Remove all services and apps from the space.

        Will leave the space itself, call delete_space() if you
        want to remove that too.

        Similar to `cf delete-space -f <space-name>`.
        """
        logging.warning("Purging all services from space %s" %
                (self.name))

        service = predix.admin.cf.services.Service()
        for service_name in self.get_instances():
            service.purge(service_name)

        apps = predix.admin.cf.apps.App()
        for app_name in self.get_apps():
            apps.delete_app(app_name)