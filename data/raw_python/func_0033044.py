def fetch_all_kernels(self):
        r"""
        Returns a generator that yields all of the kernels available to the
        droplet

        :rtype: generator of `Kernel`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        for kern in api.paginate(self.url + '/kernels', 'kernels'):
            yield Kernel(kern, doapi_manager=api)