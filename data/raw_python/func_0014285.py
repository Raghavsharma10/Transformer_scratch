def create_system(self, **system_options):
        """
        Create an OpenMM system for every supported topology file with given system options
        """
        if self.master is None:
            raise ValueError('Handler {} is not able to create systems.'.format(self))

        if isinstance(self.master, ForceField):
            system = self.master.createSystem(self.topology, **system_options)
        elif isinstance(self.master, (AmberPrmtopFile, GromacsTopFile, DesmondDMSFile)):
            system = self.master.createSystem(**system_options)
        elif isinstance(self.master, CharmmPsfFile):
            if not hasattr(self.master, 'parmset'):
                raise ValueError('PSF topology files require Charmm parameters.')
            system = self.master.createSystem(self.master.parmset, **system_options)
        else:
            raise NotImplementedError('Handler {} is not able to create systems.'.format(self))

        if self.has_box:
            system.setDefaultPeriodicBoxVectors(*self.box)
        return system