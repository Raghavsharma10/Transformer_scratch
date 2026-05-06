def _pull_branch_from_origin(self):
        """
        Pulls from origin/master, if you have unmerged conflicts
        it will raise an exception. You will need to resolve these.
        """
        try:
            ## self.repo.git.pull()
            subprocess.check_call(["git", "pull", "origin", self.branch])
        except Exception as inst:
            sys.exit("""
        Your HEAD commit conflicts with origin/{tag}. 
        Resolve, merge, and rerun versioner.py
        """)