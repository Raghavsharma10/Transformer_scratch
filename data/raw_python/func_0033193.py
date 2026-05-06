def _get_result_paths(self, data):
        """ Return a dict of ResultPath objects representing all possible output
        """
        assignment_fp = str(self.Parameters['-o'].Value).strip('"')
        if not os.path.isabs(assignment_fp):
            assignment_fp = os.path.relpath(assignment_fp, self.WorkingDir)
        return {'Assignments': ResultPath(assignment_fp, IsWritten=True)}