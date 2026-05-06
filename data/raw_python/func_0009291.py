def _create_cpe_parts(self, system, components):
        """
        Create the structure to store the input type of system associated
        with components of CPE Name (hardware, operating system and software).

        :param string system: type of system associated with CPE Name
        :param dict components: CPE Name components to store
        :returns: None
        :exception: KeyError - incorrect system
        """

        if system not in CPEComponent.SYSTEM_VALUES:
            errmsg = "Key '{0}' is not exist".format(system)
            raise ValueError(errmsg)

        elements = []
        elements.append(components)

        pk = CPE._system_and_parts[system]
        self[pk] = elements