def kafka_config(self):
        """
        kafka config
        :return:
        """
        # 读取配置文件中的端口
        config_obj = self.configure[self.args.config[1]]
        kafka_ports = config_obj.get('KAFKA_PORTS')
        # 默认端口9092
        if not kafka_ports:
            kafka_ports_arr = ['9092']
        else:
            kafka_ports_arr = kafka_ports.replace(' ', '').split(',')

        # chmod project root owner
        sudo('chown {user}:{user} -R {path}'.format(
            user=config_obj.get('user'),
            path=bigdata_conf.project_root
        ))
        # change kafka bin permission for JAVA
        sudo('chmod -R 777 {0}/bin'.format(bigdata_conf.kafka_home))

        # 配置zookeeper服务
        self.systemctl_autostart(
            'zookeeper.service',
            '/opt/kafka/bin/zookeeper-server-start.sh /opt/kafka/config/zookeeper.properties',
            '/opt/kafka/bin/zookeeper-server-stop.sh /opt/kafka/config/zookeeper.properties'
        )

        # 循环生成kafka配置文件
        with cd('{0}/config'.format(bigdata_conf.kafka_home)):
            for idx, k_port in enumerate(kafka_ports_arr):
                conf_file = 'server.properties-{0}'.format(k_port)
                run('cp server.properties {0}'.format(conf_file))

                # 修改kafka配置文件
                sed(conf_file, 'broker.id=.*', 'broker.id={0}'.format(idx))
                uncomment(conf_file, 'listeners=PLAINTEXT')
                sed(conf_file, 'PLAINTEXT://.*', 'PLAINTEXT://{0}:{1}'.format(
                    env.host_string, k_port
                ))
                sed(conf_file, 'log.dirs=.*',
                    'log.dirs=/tmp/kafka-log-{0}'.format(k_port))

                # 配置kafka服务
                self.systemctl_autostart(
                    'kafka-{0}.service'.format(k_port),
                    '/opt/kafka/bin/kafka-server-start.sh /opt/kafka/config/{0}'.format(conf_file),
                    '/opt/kafka/bin/kafka-server-stop.sh /opt/kafka/config/{0}'.format(conf_file)
                )