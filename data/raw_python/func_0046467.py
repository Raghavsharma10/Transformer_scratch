def dependencies(self, images):
        """Yield just the dependency images"""
        for dep in self.commands.dependent_images:
            if not isinstance(dep, six.string_types):
                yield dep.name

        for image, _ in self.dependency_images():
            yield image