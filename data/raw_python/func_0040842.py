def logstash_install(self):
        """
        logstash install
        :return:
        """
        with cd('/tmp'):
            if not exists('logstash.deb'):
                sudo('wget {0} -O logstash.deb'.format(
                    bigdata_conf.logstash_download_url
                ))

            sudo('dpkg -i logstash.deb')
            sudo('apt-get install -y')