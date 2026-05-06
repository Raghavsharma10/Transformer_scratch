def set_params(self, **kwargs):
        """
        set parameters in the user parameters

        these parameters are accepted:
        :param git_uri: str, uri of the git repository for the source
        :param git_ref: str, commit ID of the branch to be pulled
        :param git_branch: str, branch name of the branch to be pulled
        :param base_image: str, name of the parent image
        :param name_label: str, label of the parent image
        :param user: str, name of the user requesting the build
        :param component: str, name of the component
        :param release: str,
        :param build_image: str,
        :param build_imagestream: str,
        :param build_from: str,
        :param build_type: str, orchestrator or worker
        :param platforms: list of str, platforms to build on
        :param platform: str, platform
        :param koji_target: str, koji tag with packages used to build the image
        :param koji_task_id: str, koji ID
        :param koji_parent_build: str,
        :param koji_upload_dir: str, koji directory where the completed image will be uploaded
        :param flatpak: if we should build a Flatpak OCI Image
        :param flatpak_base_image: str, name of the Flatpack OCI Image
        :param reactor_config_map: str, name of the config map containing the reactor environment
        :param reactor_config_override: dict, data structure for reactor config to be injected as
                                        an environment variable into a worker build;
                                        when used, reactor_config_map is ignored.
        :param yum_repourls: list of str, uris of the yum repos to pull from
        :param signing_intent: bool, True to sign the resulting image
        :param compose_ids: list of int, ODCS composes to use instead of generating new ones
        :param filesystem_koji_task_id: int, Koji Task that created the base filesystem
        :param platform_node_selector: dict, a nodeselector for a user_paramsific platform
        :param scratch_build_node_selector: dict, a nodeselector for scratch builds
        :param explicit_build_node_selector: dict, a nodeselector for explicit builds
        :param auto_build_node_selector: dict, a nodeselector for auto builds
        :param isolated_build_node_selector: dict, a nodeselector for isolated builds
        :param operator_manifests_extract_platform: str, indicates which platform should upload
                                                    operator manifests to koji
        :param parent_images_digests: dict, mapping image digests to names and platforms
        """

        # Here we cater to the koji "scratch" build type, this will disable
        # all plugins that might cause importing of data to koji
        self.scratch = kwargs.get('scratch')
        # When true, it indicates build was automatically started by
        # OpenShift via a trigger, for instance ImageChangeTrigger
        self.is_auto = kwargs.pop('is_auto', False)
        # An isolated build is meant to patch a certain release and not
        # update transient tags in container registry
        self.isolated = kwargs.get('isolated')

        self.osbs_api = kwargs.pop('osbs_api', None)
        self.validate_build_variation()

        self.base_image = kwargs.get('base_image')
        self.platform_node_selector = kwargs.get('platform_node_selector', {})
        self.scratch_build_node_selector = kwargs.get('scratch_build_node_selector', {})
        self.explicit_build_node_selector = kwargs.get('explicit_build_node_selector', {})
        self.auto_build_node_selector = kwargs.get('auto_build_node_selector', {})
        self.isolated_build_node_selector = kwargs.get('isolated_build_node_selector', {})

        logger.debug("now setting params '%s' for user_params", kwargs)
        self.user_params.set_params(**kwargs)

        self.source_registry = None
        self.organization = None