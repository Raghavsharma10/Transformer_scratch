def run_with_tornado(self):
        """
        runs the tornado/websockets based test server
        """
        from zengine.tornado_server.server import runserver
        runserver(self.manager.args.addr, int(self.manager.args.port))