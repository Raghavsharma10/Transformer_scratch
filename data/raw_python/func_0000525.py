def clone(self):
        """
        Do not initialize again since everything is ready to launch app.
        :return: Initialized monitor instance
        """
        return Monitor(org=self.org, app=self.app, env=self.env)