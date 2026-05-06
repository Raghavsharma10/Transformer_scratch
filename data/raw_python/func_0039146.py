def can_be_updated(cls, dist, latest_version):
        """Determine whether package can be updated or not."""
        scheme = get_scheme('default')
        name = dist.project_name
        dependants = cls.get_dependants(name)
        for dependant in dependants:
            requires = dependant.requires()
            for requirement in cls.get_requirement(name, requires):
                req = parse_requirement(requirement)
                # Ignore error if version in requirement spec can't be parsed
                try:
                    matcher = scheme.matcher(req.requirement)
                except UnsupportedVersionError:
                    continue
                if not matcher.match(str(latest_version)):
                    return False

        return True