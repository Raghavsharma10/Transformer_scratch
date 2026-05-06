def _list_machines(self):
        """
        Request a list of all added machines.

        Populates self._machines dict with mist.client.model.Machine instances
        """
        try:
            req = self.request(self.mist_client.uri+'/clouds/'+self.id+'/machines')
            machines = req.get().json()
        except:
            # Eg invalid cloud credentials
            machines = {}

        if machines:
            for machine in machines:
                self._machines[machine['machine_id']] = Machine(machine, self)
        else:
            self._machines = {}