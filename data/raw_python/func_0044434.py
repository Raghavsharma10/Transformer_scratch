def start(self, instance_names):
        """ If start is called from the root of a Galaxy source directory with
        no args, automatically add this instance.
        """
        if not instance_names:
            configs = (os.path.join('config', 'galaxy.ini'),
                    os.path.join('config', 'galaxy.ini.sample'))
            for config in configs:
                if os.path.exists(config):
                    if not self.config_manager.is_registered(os.path.abspath(config)):
                        self.config_manager.add([config])
                    break