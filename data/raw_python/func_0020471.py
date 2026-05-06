def remove_plugin(self, phase, name, reason=None):
        """
        if config contains plugin, remove it
        """
        for p in self.template[phase]:
            if p.get('name') == name:
                self.template[phase].remove(p)
                if reason:
                    logger.info('Removing {}:{}, {}'.format(phase, name, reason))
                break