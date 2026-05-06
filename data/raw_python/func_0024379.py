def upgrade_addons_operation(self, addons_state, mode=None):
        """ Return merged set of main addons and mode's addons """
        installed = set(a.name for a in addons_state
                        if a.state in ('installed', 'to upgrade'))

        base_mode = self._get_version_mode()
        addons_list = base_mode.upgrade_addons.copy()
        if mode:
            add_mode = self._get_version_mode(mode=mode)
            addons_list |= add_mode.upgrade_addons

        to_install = addons_list - installed
        to_upgrade = installed & addons_list

        return UpgradeAddonsOperation(self.options, to_install, to_upgrade)