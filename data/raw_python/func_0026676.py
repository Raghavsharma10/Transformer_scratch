def set_preferred_names(self):
        """Choose between each entries given name and its possible aliases for
        the best one.
        """
        if len(self.entries) == 0:
            self.log.error("WARNING: `entries` is empty, loading stubs")
            self.load_stubs()

        task_str = self.get_current_task_str()
        for ni, oname in enumerate(pbar(self.entries, task_str)):
            name = self.add_entry(oname)
            self.entries[name].set_preferred_name()

            if self.args.travis and ni > self.TRAVIS_QUERY_LIMIT:
                break

        return