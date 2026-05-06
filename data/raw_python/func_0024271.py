def run(self):
        """
        read workflows, checks if it's updated,
        tries to update if there aren't any running instances of that wf
        """
        from zengine.lib.cache import WFSpecNames

        if self.manager.args.clear:
            self._clear_models()
            return

        if self.manager.args.wf_path:
            paths = self.get_wf_from_path(self.manager.args.wf_path)
        else:
            paths = self.get_workflows()

        self.count = 0

        self.do_with_submit(self.load_diagram, paths, threads=self.manager.args.threads)

        WFSpecNames().refresh()

        print("%s BPMN file loaded" % self.count)