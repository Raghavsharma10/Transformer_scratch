def _push_new_tag_to_git(self):
        """
        tags a new release and pushes to origin/master
        """
        print("Pushing new version to git")            

        ## stage the releasefile and initfileb
        subprocess.call(["git", "add", self.release_file])
        subprocess.call(["git", "add", self.init_file])
        subprocess.call([
            "git", "commit", "-m", "Updating {}/__init__.py to version {}"\
            .format(self.package, self.tag)])

        ## push changes to origin <tracked branch>
        subprocess.call(["git", "push", "origin", self.branch])

        ## create a new tag for the version number on deploy
        if self.deploy:
            subprocess.call([
                "git", "tag", "-a", self.tag,
                "-m", "Updating version to {}".format(self.tag),
                ])
            subprocess.call(["git", "push", "origin"])