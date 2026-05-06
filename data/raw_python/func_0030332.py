def dependencies_as_list(self):
        """Returns a list of dependency names."""
        dependencies = []
        for dependency in self.dependencies:
            dependencies.append(dependency.name)
        return dependencies