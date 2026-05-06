def create_container(self, conf, detach, tty):
        """Create a single container"""

        name = conf.name
        image_name = conf.image_name
        if conf.tag is not NotSpecified:
            image_name = conf.image_name_with_tag
        container_name = conf.container_name

        with conf.assumed_role():
            env = dict(e.pair for e in conf.env)

        binds = conf.volumes.binds
        command = conf.formatted_command
        volume_names = conf.volumes.volume_names
        volumes_from = list(conf.volumes.share_with_names)
        no_tty_option = conf.no_tty_option

        ports = [p.container_port.port_pair for p in conf.ports]
        port_bindings = self.exposed(conf.ports)

        uncreated = []
        for name in binds:
            if not os.path.exists(name):
                log.info("Making volume for mounting\tvolume=%s", name)
                try:
                    os.makedirs(name)
                except OSError as error:
                    uncreated.append((name, error))
        if uncreated:
            raise BadOption("Failed to create some volumes on the host", uncreated=uncreated)

        log.info("Creating container from %s\timage=%s\tcontainer_name=%s\ttty=%s", image_name, name, container_name, tty)
        if binds:
            log.info("\tUsing volumes\tvolumes=%s", volume_names)
        if env:
            log.info("\tUsing environment\tenv=%s", sorted(env.keys()))
        if ports:
            log.info("\tUsing ports\tports=%s", ports)
        if port_bindings:
            log.info("\tPort bindings: %s", port_bindings)
        if volumes_from:
            log.info("\tVolumes from: %s", volumes_from)

        host_config = conf.harpoon.docker_api.create_host_config(
              binds = binds
            , volumes_from = volumes_from
            , port_bindings = port_bindings

            , devices = conf.devices
            , lxc_conf = conf.lxc_conf
            , privileged = conf.privileged
            , restart_policy = conf.restart_policy

            , dns = conf.network.dns
            , dns_search = conf.network.dns_search
            , extra_hosts = conf.network.extra_hosts
            , network_mode = conf.network.network_mode
            , publish_all_ports = conf.network.publish_all_ports

            , cap_add = conf.cpu.cap_add
            , cap_drop = conf.cpu.cap_drop
            , mem_limit = conf.cpu.mem_limit
            , cpu_shares = conf.cpu.cpu_shares
            , cpuset_cpus = conf.cpu.cpuset_cpus
            , cpuset_mems = conf.cpu.cpuset_mems
            , memswap_limit = conf.cpu.memswap_limit

            , ulimits = conf.ulimits
            , read_only = conf.read_only_rootfs
            , log_config = conf.log_config
            , security_opt = conf.security_opt

            , **conf.other_options.host_config
            )

        container_id = conf.harpoon.docker_api.create_container(image_name
            , name=container_name
            , detach=detach
            , command=command
            , volumes=volume_names
            , environment=env

            , tty = False if no_tty_option else tty
            , user = conf.user
            , ports = ports
            , stdin_open = tty

            , hostname = conf.network.hostname
            , domainname = conf.network.domainname
            , network_disabled = conf.network.disabled

            , host_config = host_config

            , **conf.other_options.create
            )

        if isinstance(container_id, dict):
            if "errorDetail" in container_id:
                raise BadImage("Failed to create container", image=name, error=container_id["errorDetail"])
            container_id = container_id["Id"]

        return container_id