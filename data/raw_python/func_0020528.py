def create_prod_build(self, *args, **kwargs):
        """
        Create a production build

        :param git_uri: str, URI of git repository
        :param git_ref: str, reference to commit
        :param git_branch: str, branch name
        :param user: str, user name
        :param component: str, not used anymore
        :param target: str, koji target
        :param architecture: str, build architecture
        :param yum_repourls: list, URLs for yum repos
        :param koji_task_id: int, koji task ID requesting build
        :param scratch: bool, this is a scratch build
        :param platform: str, the platform name
        :param platforms: list<str>, the name of each platform
        :param release: str, the release value to use
        :param inner_template: str, name of inner template for BuildRequest
        :param outer_template: str, name of outer template for BuildRequest
        :param customize_conf: str, name of customization config for BuildRequest
        :param arrangement_version: int, numbered arrangement of plugins for orchestration workflow
        :param signing_intent: str, signing intent of the ODCS composes
        :param compose_ids: list<int>, ODCS composes used
        :return: BuildResponse instance
        """
        logger.warning("prod (all-in-one) builds are deprecated, "
                       "please use create_orchestrator_build "
                       "(support will be removed in version 0.54)")
        return self._do_create_prod_build(*args, **kwargs)