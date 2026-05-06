def dependent_images(self):
        """
        Determine the dependent images from these commands

        This includes all the FROM statements
        and any external image from a complex ADD instruction that copies from another container
        """
        found = []
        for command in self.commands:
            dep = command.dependent_image
            if dep:
                if dep not in found:
                    yield dep
                    found.append(dep)