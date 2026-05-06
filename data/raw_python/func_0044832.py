def check_candidate_exists(self, basepath, candidates):
        """
        Check that at least one candidate exist into a directory.

        Args:
            basepath (str): Directory path where to search for candidate.
            candidates (list): List of candidate file paths.

        Returns:
            list: List of existing candidates.
        """
        checked = []
        for item in candidates:
            abspath = os.path.join(basepath, item)
            if os.path.exists(abspath):
                checked.append(abspath)

        return checked