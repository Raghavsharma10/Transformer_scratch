def render_tag_from_config(self):
        """Configure tag_from_config plugin"""
        phase = 'postbuild_plugins'
        plugin = 'tag_from_config'
        if not self.has_tag_suffixes_placeholder():
            return

        unique_tag = self.user_params.image_tag.value.split(':')[-1]
        tag_suffixes = {'unique': [unique_tag], 'primary': []}

        if self.user_params.build_type.value == BUILD_TYPE_ORCHESTRATOR:
            additional_tags = self.user_params.additional_tags.value or set()

            if self.user_params.scratch.value:
                pass
            elif self.user_params.isolated.value:
                tag_suffixes['primary'].extend(['{version}-{release}'])
            elif self.user_params.tags_from_yaml.value:
                tag_suffixes['primary'].extend(['{version}-{release}'])
                tag_suffixes['primary'].extend(additional_tags)
            else:
                tag_suffixes['primary'].extend(['latest', '{version}', '{version}-{release}'])
                tag_suffixes['primary'].extend(additional_tags)

        self.pt.set_plugin_arg(phase, plugin, 'tag_suffixes', tag_suffixes)