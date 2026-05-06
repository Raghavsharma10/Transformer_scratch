def _get_existing_build_config(self, build_config):
        """
        Uses the given build config to find an existing matching build config.
        Build configs are a match if:
        - metadata.labels.git-repo-name AND metadata.labels.git-branch AND
          metadata.labels.git-full-repo are equal
        OR
        - metadata.labels.git-repo-name AND metadata.labels.git-branch are equal AND
          metadata.spec.source.git.uri are equal
        OR
        - metadata.name are equal
        """

        bc_labels = build_config['metadata']['labels']
        git_labels = {
            "label_selectors": [(key, bc_labels[key]) for key in self._GIT_LABEL_KEYS]
        }
        old_labels_kwargs = {
            "label_selectors": [(key, bc_labels[key]) for key in self._OLD_LABEL_KEYS],
            "filter_key": FILTER_KEY,
            "filter_value": graceful_chain_get(build_config, *FILTER_KEY.split('.'))
        }
        name = {
            "build_config_id": build_config['metadata']['name']
        }

        queries = (
            (self.os.get_build_config_by_labels, git_labels),
            (self.os.get_build_config_by_labels_filtered, old_labels_kwargs),
            (self.os.get_build_config, name),
        )

        existing_bc = None
        for func, kwargs in queries:
            try:
                existing_bc = func(**kwargs)
                # build config found
                break
            except OsbsException as exc:
                # doesn't exist
                logger.info('Build config NOT found via %s: %s',
                            func.__name__, str(exc))
                continue

        return existing_bc