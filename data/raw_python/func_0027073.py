def get_resources_count(self, link):
        """
        Count total number of all resources connected to link
        """
        total = 0
        for model in SupportedServices.get_service_resources(link.service):
            # Format query path from resource to service project link
            query = {model.Permissions.project_path.split('__')[0]: link}
            total += model.objects.filter(**query).count()
        return total