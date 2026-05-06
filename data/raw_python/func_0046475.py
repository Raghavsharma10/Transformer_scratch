def make_context(self, docker_file=None):
        """Determine the docker lines for this image"""
        kwargs = {"silent_build": self.harpoon.silent_build, "extra_context": self.commands.extra_context}
        if docker_file is None:
            docker_file = self.docker_file
        with ContextBuilder().make_context(self.context, **kwargs) as ctxt:
            self.add_docker_file_to_tarfile(docker_file, ctxt.t)
            yield ctxt