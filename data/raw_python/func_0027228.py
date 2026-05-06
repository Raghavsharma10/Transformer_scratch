def stats(self, request, uuid=None):
        """
        This endpoint returns allocation of resources for current service setting.
        Answer is service-specific dictionary. Example output for OpenStack:

        * vcpu - maximum number of vCPUs (from hypervisors)
        * vcpu_quota - maximum number of vCPUs(from quotas)
        * vcpu_usage - current number of used vCPUs

        * ram - total size of memory for allocation (from hypervisors)
        * ram_quota - maximum number of memory (from quotas)
        * ram_usage - currently used memory size on all physical hosts

        * storage - total available disk space on all physical hosts (from hypervisors)
        * storage_quota - maximum number of storage (from quotas)
        * storage_usage - currently used storage on all physical hosts

        {
            'vcpu': 10,
            'vcpu_quota': 7,
            'vcpu_usage': 5,
            'ram': 1000,
            'ram_quota': 700,
            'ram_usage': 500,
            'storage': 10000,
            'storage_quota': 7000,
            'storage_usage': 5000
        }
        """

        service_settings = self.get_object()
        backend = service_settings.get_backend()

        try:
            stats = backend.get_stats()
        except ServiceBackendNotImplemented:
            stats = {}

        return Response(stats, status=status.HTTP_200_OK)