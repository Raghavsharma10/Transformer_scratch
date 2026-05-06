def versions(self) -> List(BlenderVersion):
        """
        The versions associated with Blender
        """

        return [BlenderVersion(tag) for tag in self.git_repo.tags] + [BlenderVersion(BLENDER_VERSION_MASTER)]