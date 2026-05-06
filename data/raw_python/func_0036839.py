def apply_config(self, config):
        """
        Constructs HAProxyConfig and HAProxyControl instances based on the
        contents of the config.

        This is mostly a matter of constructing the configuration stanzas.
        """
        self.haproxy_config_path = config["config_file"]

        global_stanza = Stanza("global")
        global_stanza.add_lines(config.get("global", []))
        global_stanza.add_lines([
            "stats socket %s mode 600 level admin" % config["socket_file"],
            "stats timeout 2m"
        ])

        defaults_stanza = Stanza("defaults")
        defaults_stanza.add_lines(config.get("defaults", []))

        proxy_stanzas = [
            ProxyStanza(
                name, proxy["port"], proxy["upstreams"],
                proxy.get("options", []),
                proxy.get("bind_address")
            )
            for name, proxy in six.iteritems(config.get("proxies", {}))
        ]

        stats_stanza = None
        if "stats" in config:
            stats_stanza = StatsStanza(
                config["stats"]["port"], config["stats"].get("uri", "/")
            )
            for timeout in ("client", "connect", "server"):
                if timeout in config["stats"].get("timeouts", {}):
                    stats_stanza.add_line(
                        "timeout %s %d" % (
                            timeout,
                            config["stats"]["timeouts"][timeout]
                        )
                    )

        self.config_file = HAProxyConfig(
            global_stanza, defaults_stanza,
            proxy_stanzas=proxy_stanzas, stats_stanza=stats_stanza,
            meta_clusters=config.get("meta_clusters", {}),
            bind_address=config.get("bind_address")
        )

        self.control = HAProxyControl(
            config["config_file"], config["socket_file"], config["pid_file"],
        )