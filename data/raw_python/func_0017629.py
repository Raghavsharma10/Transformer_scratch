def run(self):
        """
        Calls the main function of a plugin and mutates the output dict
        with its return value. Provides an easy way to change the output
        whilst not needing to constantly poll a queue in another thread and
        allowing plugin's to manage their own intervals.
        """
        self.running = True
        while self.running:
            ret = self.func()
            self.output_dict[ret['name']] = ret
            time.sleep(self.interval)
        return