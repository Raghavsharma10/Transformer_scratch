def get_tuid(self, branch, revision, file):
        """
        :param branch: BRANCH TO FIND THE REVISION/FILE
        :param revision: THE REVISION NUNMBER
        :param file: THE FULL PATH TO A SINGLE FILE
        :return: A LIST OF TUIDS
        """
        service_response = wrap(self.get_tuids(branch, revision, [file]))
        for f, t in service_response.items():
            return t