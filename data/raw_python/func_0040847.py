def spark_config(self):
        """
        config spark
        :return:
        """
        configs = [
            'export LD_LIBRARY_PATH={0}/lib/native/:$LD_LIBRARY_PATH'.format(
                bigdata_conf.hadoop_home
            ),
            'export SPARK_LOCAL_IP={0}'.format(env.host_string)
        ]

        append(bigdata_conf.global_env_home, configs, use_sudo=True)
        run('source {0}'.format(bigdata_conf.global_env_home))