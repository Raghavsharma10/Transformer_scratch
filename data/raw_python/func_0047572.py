def run(self):
        """Run checker."""

        def split(module):
            """Split module into submodules."""
            return tuple(module.split("."))

        def modcmp(lib=(), test=()):
            """Compare import modules."""
            if len(lib) > len(test):
                return False
            return all(a == b for a, b in zip(lib, test))

        mods_1st_party = set()
        mods_3rd_party = set()

        # Get 1st party modules (used for absolute imports).
        modules = [project2module(self.setup.keywords.get('name', ""))]
        if modules[0] in self.known_modules:
            modules = self.known_modules[modules[0]]
        mods_1st_party.update(split(x) for x in modules)

        requirements = self.requirements
        if self.setup.redirected:
            # Use requirements from setup if available.
            requirements = self.setup.get_requirements(
                setup=self.processing_setup_py,
                tests=True,
            )

        # Get 3rd party module names based on requirements.
        for requirement in requirements:
            modules = [project2module(requirement.project_name)]
            if modules[0] in KNOWN_3RD_PARTIES:
                modules = KNOWN_3RD_PARTIES[modules[0]]
            if modules[0] in self.known_modules:
                modules = self.known_modules[modules[0]]
            mods_3rd_party.update(split(x) for x in modules)

        # When processing setup.py file, forcefully add setuptools to the
        # project requirements. Setuptools might be required to build the
        # project, even though it is not listed as a requirement - this
        # package is required to run setup.py, so listing it as a setup
        # requirement would be pointless.
        if self.processing_setup_py:
            mods_3rd_party.add(split("setuptools"))

        for node in ImportVisitor(self.tree).imports:
            _mod = split(node.mod)
            _alt = split(node.alt)
            if any([_mod[0] == x for x in STDLIB]):
                continue
            if any([modcmp(x, _mod) or modcmp(x, _alt)
                    for x in mods_1st_party]):
                continue
            if any([modcmp(x, _mod) or modcmp(x, _alt)
                    for x in mods_3rd_party]):
                continue
            yield (
                node.line,
                node.offset,
                ERRORS['I900'].format(pkg=node.mod),
                Flake8Checker,
            )