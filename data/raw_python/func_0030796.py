def get_dependency_structure(self, artifact=None, include_dependencies=False):
        """
        Reads dependency structure. If an artifact is passed in you get only its dependencies otherwise the complete
        structure is returned.

        :param artifact: an artifact task or artifact name if only an artifact's deps are needed
        :param include_dependencies: flag to include also dependencies in returned artifacts and their dependencies in
                                     dependencies dict
        :return: tuple of artifact names list and dependencies dictionary where value is a Task list
        """
        artifacts = []
        dependencies_dict = {}
        if artifact:
            if isinstance(artifact, str):
                artifact = self.get_tasks().get_task(artifact)
            artifacts.append(artifact.name)
            dependencies_dict[artifact.name] =  artifact.ordered_dependencies()

            if include_dependencies:
                for dep in dependencies_dict[artifact.name]:
                    artifacts.append(dep.name)
                    dependencies_dict[dep.name] =  dep.ordered_dependencies()
        else:
            for key, task in self.get_tasks().tasks.iteritems():
                artifacts.append(task.name)
                dependencies_dict[task.name] = task.ordered_dependencies()

        return artifacts, dependencies_dict