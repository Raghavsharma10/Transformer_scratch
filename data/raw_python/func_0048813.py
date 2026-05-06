def external_dependencies(self):
        """
        Return all the external images this Dockerfile will depend on

        These are images from self.dependent_images that aren't defined in this configuration.
        """
        found = []
        for dep in self.dependent_images:
            if isinstance(dep, six.string_types):
                if dep not in found:
                    yield dep
                    found.append(dep)