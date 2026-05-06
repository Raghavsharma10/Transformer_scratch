def run_deps(self, conf, images):
        """Start containers for all our dependencies"""
        for dependency_name, detached in conf.dependency_images(for_running=True):
            try:
                self.run_container(images[dependency_name], images, detach=detached, dependency=True)
            except Exception as error:
                raise BadImage("Failed to start dependency container", image=conf.name, dependency=dependency_name, error=error)