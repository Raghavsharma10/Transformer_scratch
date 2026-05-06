def kafka_install(self):
        """
        kafka download and install
        :return:
        """
        with cd('/tmp'):
            if not exists('kafka.tgz'):
                sudo('wget {0} -O kafka.tgz'.format(
                    bigdata_conf.kafka_download_url
                ))

            sudo('tar -zxf kafka.tgz')

            sudo('rm -rf {0}'.format(bigdata_conf.kafka_home))
            sudo('mv kafka_* {0}'.format(bigdata_conf.kafka_home))