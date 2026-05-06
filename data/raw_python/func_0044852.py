def remove_post_process(self, name):
        """remove a post-process

        Parameters
        ----------
        name : str
            name of the post-process to remove.
        """
        self._pprocesses = [post_process
                            for post_process in self._pprocesses
                            if post_process.name != name]