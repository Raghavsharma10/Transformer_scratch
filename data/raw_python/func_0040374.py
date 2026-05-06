def start_agent(self, cfgin=True):
        """
        CLI interface to start 12-factor service
        """

        default_conf = {
            "threads": {
                "result": {
                    "number": 0,
                    "function": None
                },
                "worker": {
                    "number": 0,
                    "function": None
                },
            },
            "interval": {
                "refresh": 900,
                "heartbeat": 300,
                "reporting": 300,
                "test": 60
            },
            "heartbeat-hook": False
        }
        indata = {}
        if cfgin:
            indata = json.load(sys.stdin)
        elif os.environ.get("REFLEX_MONITOR_CONFIG"):
            indata = os.environ.get("REFLEX_MONITOR_CONFIG")
            if indata[0] != "{":
                indata = base64.b64decode(indata)
        else:
            self.NOTIFY("Using default configuration")

        conf = dictlib.union(default_conf, indata)

        conf['threads']['result']['function'] = self.handler_thread
        conf['threads']['worker']['function'] = self.worker_thread

        self.NOTIFY("Starting monitor Agent")
        try:
            self.configure(conf).start()
        except KeyboardInterrupt:
            self.thread_stopper.set()
            if self.refresh_stopper:
                self.refresh_stopper.set()
            if self.heartbeat_stopper:
                self.heartbeat_stopper.set()
            if self.reporting_stopper:
                self.reporting_stopper.set()