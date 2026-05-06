def finalize(self):
        """Output the number of instances that contained dead code."""
        if self.total_instances > 1:
            print('{} of {} instances contained dead code.'
                  .format(self.dead_code_instances, self.total_instances))