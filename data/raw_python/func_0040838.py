def java_install(self):
        """
        install java
        :return:
        """
        sudo('apt-get install openjdk-8-jdk -y')

        java_home = run('readlink -f /usr/bin/java | '
                        'sed "s:/jre/bin/java::"')

        append(bigdata_conf.global_env_home, 'export JAVA_HOME={0}'.format(
            java_home
        ), use_sudo=True)
        run('source {0}'.format(bigdata_conf.global_env_home))