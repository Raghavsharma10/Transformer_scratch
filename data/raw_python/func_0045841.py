def stage_run_intervention(self, conf, just_do_it=False):
        """Start an intervention!"""
        if not conf.harpoon.interactive or conf.harpoon.no_intervention:
            return

        if just_do_it:
            answer = 'y'
        else:
            hp.write_to(conf.harpoon.stdout, "!!!!\n")
            hp.write_to(conf.harpoon.stdout, "Failed to run the container!\n")
            hp.write_to(conf.harpoon.stdout, "Do you want commit the container in it's current state and {0} into it to debug?\n".format(conf.resolved_shell))
            conf.harpoon.stdout.flush()
            answer = input("[y]: ")
        if not answer or answer.lower().startswith("y"):
            with self.commit_and_run(conf.container_id, conf, command=conf.resolved_shell):
                pass