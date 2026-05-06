def intervention(self, commit, conf):
        """Ask the user if they want to commit this container and run sh in it"""
        if not conf.harpoon.interactive or conf.harpoon.no_intervention:
            yield
            return

        hp.write_to(conf.harpoon.stdout, "!!!!\n")
        hp.write_to(conf.harpoon.stdout, "It would appear building the image failed\n")
        hp.write_to(conf.harpoon.stdout, "Do you want to run {0} where the build to help debug why it failed?\n".format(conf.resolved_shell))
        conf.harpoon.stdout.flush()
        answer = input("[y]: ")
        if answer and not answer.lower().startswith("y"):
            yield
            return

        with self.commit_and_run(commit, conf, command=conf.resolved_shell):
            yield