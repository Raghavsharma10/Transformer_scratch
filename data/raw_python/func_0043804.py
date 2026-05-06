def run_update(self, template_name=None, service_dir=None):
        " Run update scripts. "

        LOGGER.info('Site Update start.')
        print_header('Update %s' % self.get_name())
        map(call, self._gen_scripts(
            'update', template_name=template_name, service_dir=service_dir))
        LOGGER.info('Site Update done.')
        return True