def run(self):
        """
        Starts a development server for the zengine application
        """
        print("Development server started on http://%s:%s. \n\nPress Ctrl+C to stop\n" % (
            self.manager.args.addr,
            self.manager.args.port)
              )
        if self.manager.args.server_type == 'falcon':
            self.run_with_falcon()
        elif self.manager.args.server_type == 'tornado':
            self.run_with_tornado()