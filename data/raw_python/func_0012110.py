def shutdown(self):
        """Executed on shutdown of application"""
        for wsock in self.wsocks:
            wsock.shutdown()
        for api in self.apis:
            api.shutdown()