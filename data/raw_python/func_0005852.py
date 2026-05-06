def configure(self):
        """Configures broodlord mode and returns emperor and zerg sections.

        :rtype: tuple
        """
        section_emperor = self.section_emperor
        section_zerg = self.section_zerg

        socket = self.socket

        section_emperor.workers.set_zerg_server_params(socket=socket)
        section_emperor.empire.set_emperor_params(vassals_home=self.vassals_home)
        section_emperor.empire.set_mode_broodlord_params(**self.broodlord_params)

        section_zerg.name = 'zerg'
        section_zerg.workers.set_zerg_client_params(server_sockets=socket)

        if self.die_on_idle:
            section_zerg.master_process.set_idle_params(timeout=30, exit=True)

        return section_emperor, section_zerg