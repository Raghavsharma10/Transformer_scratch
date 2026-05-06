def set_up(self):
        """Set up your applications and the test environment."""
        self.path.state = self.path.gen.joinpath("state")
        if self.path.state.exists():
            self.path.state.rmtree(ignore_errors=True)
        self.path.state.mkdir()

        if self.path.gen.joinpath("q").exists():
            self.path.gen.joinpath("q").remove()

        for filename, text in self.given.get("files", {}).items():
            filepath = self.path.state.joinpath(filename)
            if not filepath.dirname().exists():
                filepath.dirname().makedirs()
            filepath.write_text(text)

        for filename, linkto in self.given.get("symlinks", {}).items():
            filepath = self.path.state.joinpath(filename)
            linktopath = self.path.state.joinpath(linkto)
            linktopath.symlink(filepath)

        for filename, permission in self.given.get("permissions", {}).items():
            filepath = self.path.state.joinpath(filename)
            filepath.chmod(int(permission, 8))


        pylibrary = hitchbuildpy.PyLibrary(
            name="py3.5.0",
            base_python=hitchbuildpy.PyenvBuild("3.5.0").with_build_path(self.path.share),
            module_name="pathquery",
            library_src=self.path.project,
        ).with_build_path(self.path.gen)
        
        pylibrary.ensure_built()
        
        self.python = pylibrary.bin.python


        self.example_py_code = ExamplePythonCode(self.python, self.path.state)\
            .with_code(self.given.get('code', ''))\
            .with_setup_code(self.given.get('setup', ''))\
            .with_terminal_size(160, 100)\
            .with_env(TMPDIR=self.path.gen)\
            .with_long_strings(
                yaml_snippet_1=self.given.get('yaml_snippet_1'),
                yaml_snippet=self.given.get('yaml_snippet'),
                yaml_snippet_2=self.given.get('yaml_snippet_2'),
                modified_yaml_snippet=self.given.get('modified_yaml_snippet'),
            )