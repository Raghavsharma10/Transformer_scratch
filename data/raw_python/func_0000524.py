def launch(self, timeout=2):
        """
        Hierapp instance, with environment dependencies:
        - can be launched within short timeout
        - auto-destroys shortly
        """
        self.start_time = time.time()
        self.end_time = time.time()
        instance = self.app.launch(environment=self.env)
        time.sleep(2) # Instance need time to appear in ui

        assert instance.running(timeout=timeout), "Monitor didn't get Active state"
        launched = instance.status == 'Active'
        instance.reschedule_workflow(workflow_name='destroy', timestamp=self.destroy_interval)
        assert instance.destroyed(timeout=timeout), "Monitor didn't get Destroyed after short time"
        stopped = instance.status == 'Destroyed'
        instance.force_remove()
        self.end_time = time.time()
        self.status = launched and stopped