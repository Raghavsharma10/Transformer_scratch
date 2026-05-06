def machines(self, id=None, name=None, search=None):
        """
        Property-like function to call the _list_machines function in order to populate self._machines dict

        :returns: A list of Machine instances.
        """
        if self._machines is None:
            self._machines = {}
            self._list_machines()

        if id:
            return [self._machines[machine_id] for machine_id in self._machines.keys()
                    if str(id) == str(self._machines[machine_id].id)]
        elif name:
            return [self._machines[machine_id] for machine_id in self._machines.keys()
                    if name == self._machines[machine_id].name]
        elif search:
            return [self._machines[machine_id] for machine_id in self._machines.keys()
                    if str(search) == str(self._machines[machine_id].name)
                    or str(search) == str(self._machines[machine_id].id)]
        else:
            return [self._machines[machine_id] for machine_id in self._machines.keys()]