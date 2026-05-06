def taskdir(self):
        """ Return the directory under which all artefacts are stored. """
        return os.path.join(self.BASE, self.TAG, self.task_family)