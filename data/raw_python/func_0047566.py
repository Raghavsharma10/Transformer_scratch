def get_requirements(
            self, install=True, extras=True, setup=False, tests=False):
        """Get package requirements."""
        requires = []
        if install:
            requires.extend(parse_requirements(
                self.keywords.get('install_requires', ()),
            ))
        if extras:
            for r in self.keywords.get('extras_require', {}).values():
                requires.extend(parse_requirements(r))
        if setup:
            requires.extend(parse_requirements(
                self.keywords.get('setup_requires', ()),
            ))
        if tests:
            requires.extend(parse_requirements(
                self.keywords.get('tests_require', ()),
            ))
        return requires