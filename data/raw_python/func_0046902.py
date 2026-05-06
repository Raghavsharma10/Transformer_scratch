def list_machines(self):
        """Retrieve a list of machines in the fleet cluster

        Yields:
            Machine: The next machine in the cluster

        Raises:
            fleet.v1.errors.APIError: Fleet returned a response code >= 400

        """
        # loop through each page of results
        for page in self._request('Machines.List'):
            # return each machine in the current page
            for machine in page.get('machines', []):
                yield Machine(data=machine)