def start_container(self, conf, tty=True, detach=False, is_dependency=False, no_intervention=False):
        """Start up a single container"""
        # Make sure we can bind to our specified ports!
        if not conf.harpoon.docker_api.base_url.startswith("http"):
            self.find_bound_ports(conf.ports)

        container_id = conf.container_id
        container_name = conf.container_name

        conf.harpoon.network_manager.register(conf, container_name)

        log.info("Starting container %s (%s)", container_name, container_id)

        try:
            if not detach and not is_dependency:
                self.start_tty(conf, interactive=tty, **conf.other_options.start)
            else:
                conf.harpoon.docker_api.start(container_id
                    , **conf.other_options.start
                    )
        except docker.errors.APIError as error:
            if str(error).startswith("404 Client Error: Not Found"):
                log.error("Container died before we could even get to it...")

        inspection = None
        if not detach and not is_dependency:
            inspection = self.get_exit_code(conf)

        if inspection and not no_intervention:
            if not inspection["State"]["Running"] and inspection["State"]["ExitCode"] != 0:
                self.stage_run_intervention(conf)
                raise BadImage("Failed to run container", container_id=container_id, container_name=container_name, reason="nonzero exit code after launch")

        if not is_dependency and conf.harpoon.intervene_afterwards and not no_intervention:
            self.stage_run_intervention(conf, just_do_it=True)