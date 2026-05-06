def remove_tag_and_push_registries(tag_and_push_registries, version):
        """
        Remove matching entries from tag_and_push_registries (in-place)

        :param tag_and_push_registries: dict, uri -> dict
        :param version: str, 'version' to match against
        """
        registries = [uri
                      for uri, regdict in tag_and_push_registries.items()
                      if regdict['version'] == version]
        for registry in registries:
            logger.info("removing %s registry: %s", version, registry)
            del tag_and_push_registries[registry]