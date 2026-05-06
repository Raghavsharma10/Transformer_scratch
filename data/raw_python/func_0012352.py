def parse_package_ref(self, ref):
        """
        Return tuple of architecture, package_name, version, id
        """
        if not ref:
            return None
        parsed = re.match('(.*)\ (.*)\ (.*)\ (.*)', ref)
        return parsed.groups()