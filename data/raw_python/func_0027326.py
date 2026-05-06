def project_list(self):
        """The list of :py:class:`pylsdj.Project` s that the
        .sav file contains"""
        return [(i, self.projects[i]) for i in sorted(self.projects.keys())]