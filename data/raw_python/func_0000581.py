def run_payload(self, payload, *, flavour: ModuleType):
        """Execute one payload after its runner is started and return its output"""
        return self.runners[flavour].run_payload(payload)