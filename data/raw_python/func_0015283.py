def install(self, struct, ui, debug=False):
        """
        This is the only method that should be called from outside. Call it
        like:
        `DependencyInstaller(struct)` and it will install packages which are
        not present on system (it uses package managers specified by `struct`
        structure)
        """
        # the system dependencies should always go first
        self.__add_dependencies(self.get_system_deptype_shortcut(), [])
        for dep_dict in struct:
            for dep_t, dep_l in dep_dict.items():
                self._process_dependency(dep_t, dep_l)
        if self.dependencies:
            self._install_dependencies(ui, debug)