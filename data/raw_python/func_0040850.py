def add_spark_slave(self, master, slave, configure):
        """
        add spark slave
        :return:
        """
        # go to master server, add config
        self.reset_server_env(master, configure)
        with cd(bigdata_conf.spark_home):
            if not exists('conf/spark-env.sh'):
                sudo('cp conf/spark-env.sh.template conf/spark-env.sh')

            spark_env = bigdata_conf.spark_env.format(
                spark_home=bigdata_conf.spark_home,
                hadoop_home=bigdata_conf.hadoop_home,
                host=env.host_string,
                SPARK_WORKER_MEMORY=configure[master].get(
                    'SPARK_WORKER_MEMORY', '512M'
                )
            )

            put(StringIO(spark_env), 'conf/spark-env.sh', use_sudo=True)

            if not exists('conf/slaves'):
                sudo('cp conf/slaves.template conf/slaves')

        # comment slaves localhost
        comment('{0}/conf/slaves'.format(bigdata_conf.spark_home),
                'localhost', use_sudo=True)

        # add slave into config
        append('{0}/conf/slaves'.format(bigdata_conf.spark_home),
               '\n{0}'.format(configure[slave]['host']), use_sudo=True)

        run('scp -r {0} {1}@{2}:/opt'.format(
            bigdata_conf.spark_home,
            configure[slave]['user'],
            configure[slave]['host']
        ))

        # go to slave server
        self.reset_server_env(slave, configure)

        append(bigdata_conf.global_env_home, 'export SPARK_LOCAL_IP={0}'.format(
            configure[slave]['host']
        ), use_sudo=True)
        run('source {0}'.format(bigdata_conf.global_env_home))

        # go to master server, restart server
        self.reset_server_env(master, configure)
        with cd(bigdata_conf.spark_home):
            run('./sbin/stop-master.sh')
            run('./sbin/stop-slaves.sh')
            run('./sbin/start-master.sh')
            run('./sbin/start-slaves.sh')