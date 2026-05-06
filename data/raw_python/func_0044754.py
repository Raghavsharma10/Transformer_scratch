def __get_supervisor(self):
        """ Return the supervisor proxy object

        Should probably use this more rather than supervisorctl directly
        """
        options = supervisorctl.ClientOptions()
        options.realize(args=['-c', self.supervisord_conf_path])
        return supervisorctl.Controller(options).get_supervisor()