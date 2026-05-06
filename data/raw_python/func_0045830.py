def stop_deps(self, conf, images):
        """Stop the containers for all our dependencies"""
        for dependency, _ in conf.dependency_images():
            self.stop_deps(images[dependency], images)
            try:
                self.stop_container(images[dependency], fail_on_bad_exit=True, fail_reason="Failed to run dependency container")
            except BadImage:
                raise
            except Exception as error:
                log.warning("Failed to stop dependency container\timage=%s\tdependency=%s\tcontainer_name=%s\terror=%s", conf.name, dependency, images[dependency].container_name, error)