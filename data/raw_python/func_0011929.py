def run(self):
        """Executed on startup of application"""
        self.api = self.context.get("cls")(self.context)
        self.context["inst"].append(self)  # Adapters used by strategies

        for call, calldata in self.context.get("calls", {}).items():
            def loop():
                """Loop on event scheduler, calling calls"""
                while not self.stopped.wait(calldata.get("delay", None)):
                    self.call(call, calldata.get("arguments", None))

            self.thread[call] = Process(target=loop)
            self.thread[call].start()