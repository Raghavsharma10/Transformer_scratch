def run_activity(self):
        """
        runs the method that referenced from current task
        """
        activity = self.current.activity
        if activity:
            if activity not in self.wf_activities:
                self._load_activity(activity)
            self.current.log.debug(
                "Calling Activity %s from %s" % (activity, self.wf_activities[activity]))
            self.wf_activities[self.current.activity](self.current)