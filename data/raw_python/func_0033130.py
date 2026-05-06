def _get_result_paths(self, data):
        """ Set the result paths
        """

        # Swarm OTU map (mandatory output)
        return {'OtuMap': ResultPath(Path=self.Parameters['-o'].Value,
                                     IsWritten=True)}