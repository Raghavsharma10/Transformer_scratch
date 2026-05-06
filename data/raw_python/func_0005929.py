def mutate(self):
        """Mutates current section."""
        section = self.section
        project_name = self.project_name

        section.project_name = project_name

        self.contribute_runtime_dir()

        main = section.main_process
        main.set_naming_params(prefix='[%s] ' % project_name)

        main.set_pid_file(
            self.get_pid_filepath(),
            before_priv_drop=False,  # For vacuum to cleanup properly.
            safe=True,
        )

        section.master_process.set_basic_params(
            fifo_file=self.get_fifo_filepath(),
        )

        # todo maybe autoreload in debug

        apps = section.applications
        apps.set_basic_params(
            manage_script_name=True,
        )

        self.contribute_error_pages()
        self.contribute_static()