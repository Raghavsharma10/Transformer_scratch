def render_tag_from_config(self):
        """Configure tag_from_config plugin"""
        phase = 'postbuild_plugins'
        plugin = 'tag_from_config'
        if not self.has_tag_suffixes_placeholder():
            return

        unique_tag = self.spec.image_tag.value.split(':')[-1]
        tag_suffixes = {'unique': [unique_tag], 'primary': []}

        if self.spec.build_type.value == BUILD_TYPE_ORCHESTRATOR:
            if self.scratch:
                pass
            elif self.isolated:
                tag_suffixes['primary'].extend(['{version}-{release}'])
            elif self._repo_info.additional_tags.from_container_yaml:
                tag_suffixes['primary'].extend(['{version}-{release}'])
                tag_suffixes['primary'].extend(self._repo_info.additional_tags.tags)
            else:
                tag_suffixes['primary'].extend(['latest', '{version}', '{version}-{release}'])
                tag_suffixes['primary'].extend(self._repo_info.additional_tags.tags)

        self.dj.dock_json_set_arg(phase, plugin, 'tag_suffixes', tag_suffixes)