def set_params(self, **kwargs):
        """
        set parameters according to specification

        these parameters are accepted:

        :param pulp_secret: str, resource name of pulp secret
        :param koji_target: str, koji tag with packages used to build the image
        :param kojiroot: str, URL from which koji packages are fetched
        :param kojihub: str, URL of the koji hub
        :param koji_certs_secret: str, resource name of secret that holds the koji certificates
        :param koji_task_id: int, Koji Task that created this build config
        :param flatpak: if we should build a Flatpak OCI Image
        :param filesystem_koji_task_id: int, Koji Task that created the base filesystem
        :param pulp_registry: str, name of pulp registry in dockpulp.conf
        :param sources_command: str, command used to fetch dist-git sources
        :param architecture: str, architecture we are building for
        :param vendor: str, vendor name
        :param build_host: str, host the build will run on or None for auto
        :param authoritative_registry: str, the docker registry authoritative for this image
        :param distribution_scope: str, distribution scope for this image
                                   (private, authoritative-source-only, restricted, public)
        :param use_auth: bool, use auth from atomic-reactor?
        :param platform_node_selector: dict, a nodeselector for a specific platform
        :param platform_descriptors: dict, platforms and their archiectures and enable_v1 settings
        :param scratch_build_node_selector: dict, a nodeselector for scratch builds
        :param explicit_build_node_selector: dict, a nodeselector for explicit builds
        :param auto_build_node_selector: dict, a nodeselector for auto builds
        :param isolated_build_node_selector: dict, a nodeselector for isolated builds
        :param is_auto: bool, indicates if build is auto build
        :param parent_images_digests: dict, mapping image names with tags to platform specific
                                      digests, example:
                                      {'registry.fedorahosted.org/fedora:29': {
                                          x86_64': 'registry.fedorahosted.org/fedora@sha256:....'}
                                      }
        """

        # Here we cater to the koji "scratch" build type, this will disable
        # all plugins that might cause importing of data to koji
        self.scratch = kwargs.pop('scratch', False)
        # When true, it indicates build was automatically started by
        # OpenShift via a trigger, for instance ImageChangeTrigger
        self.is_auto = kwargs.pop('is_auto', False)
        # An isolated build is meant to patch a certain release and not
        # update transient tags in container registry
        self.isolated = kwargs.pop('isolated', False)

        self.validate_build_variation()

        self.base_image = kwargs.get('base_image')
        self.platform_node_selector = kwargs.get('platform_node_selector', {})
        self.platform_descriptors = kwargs.get('platform_descriptors', {})
        self.scratch_build_node_selector = kwargs.get('scratch_build_node_selector', {})
        self.explicit_build_node_selector = kwargs.get('explicit_build_node_selector', {})
        self.auto_build_node_selector = kwargs.get('auto_build_node_selector', {})
        self.isolated_build_node_selector = kwargs.get('isolated_build_node_selector', {})

        logger.debug("setting params '%s' for %s", kwargs, self.spec)
        self.spec.set_params(**kwargs)
        self.osbs_api = kwargs.pop('osbs_api')