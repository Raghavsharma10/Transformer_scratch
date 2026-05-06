def run_linter(self, linter) -> None:
        """Run a checker class"""
        self.current = linter.name

        if (linter.name not in self.parser["all"].as_list("linters")
                or linter.base_pyversion > sys.version_info):  # noqa: W503
            return

        if any(x not in self.installed for x in linter.requires_install):
            raise ModuleNotInstalled(linter.requires_install)

        linter.add_output_hook(self.out_func)
        linter.set_config(self.fn, self.parser[linter.name])
        linter.run(self.files)
        self.status_code = self.status_code or linter.status_code