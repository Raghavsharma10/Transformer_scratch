def _find_all_step_methods(self):
        """
        Finds all _step<n> methods where n is an integer in this class.
        """
        steps = ([method for method in dir(self) if callable(getattr(self, method)) and
                  re.match(r'_step\d+\d+.*', method)])
        steps = sorted(steps)
        for step in steps:
            self._steps.append(getattr(self, step))