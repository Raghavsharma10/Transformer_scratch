def count(self, request):
        """
        Count resources by type. Example output:

        .. code-block:: javascript

            {
                "Amazon.Instance": 0,
                "GitLab.Project": 3,
                "Azure.VirtualMachine": 0,
                "DigitalOcean.Droplet": 0,
                "OpenStack.Instance": 0,
                "GitLab.Group": 8
            }
        """
        queryset = self.filter_queryset(self.get_queryset())
        return Response({SupportedServices.get_name_for_model(qs.model): qs.count()
                         for qs in queryset.querysets})