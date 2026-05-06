def render_name(self, name, image_tag, platform):
        """Sets the Build/BuildConfig object name"""

        if self.scratch or self.isolated:
            name = image_tag
            # Platform name may contain characters not allowed by OpenShift.
            if platform:
                platform_suffix = '-{}'.format(platform)
                if name.endswith(platform_suffix):
                    name = name[:-len(platform_suffix)]

            _, salt, timestamp = name.rsplit('-', 2)

            if self.scratch:
                name = 'scratch-{}-{}'.format(salt, timestamp)
            elif self.isolated:
                name = 'isolated-{}-{}'.format(salt, timestamp)

        # !IMPORTANT! can't be too long: https://github.com/openshift/origin/issues/733
        self.template['metadata']['name'] = name