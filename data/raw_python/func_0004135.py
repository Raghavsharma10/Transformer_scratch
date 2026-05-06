def create_from_root(self, root_source):
        """Return a populated Object Root from dictionnary datas
        """

        root_dto = ObjectRoot()

        root_dto.configuration = root_source.configuration
        root_dto.versions = [Version(x) for x in root_source.versions.values()]

        for version in sorted(root_source.versions.values()):
            hydrator = Hydrator(version, root_source.versions, root_source.versions[version.name].types)
            for method in version.methods.values():
                hydrator.hydrate_method(root_dto, root_source, method)
            for type in version.types.values():
                hydrator.hydrate_type(root_dto, root_source, type)

        self.define_changes_status(root_dto)

        return root_dto