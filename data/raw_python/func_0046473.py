def build_and_run(self, images):
        """Make this image and run it"""
        from harpoon.ship.builder import Builder
        Builder().make_image(self, images)

        try:
            Runner().run_container(self, images)
        except DockerAPIError as error:
            raise BadImage("Failed to start the container", error=error)