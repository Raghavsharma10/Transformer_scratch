def hadoop_install(self):
        """
        install hadoop
        :return:
        """
        with cd('/tmp'):
            if not exists('hadoop.tar.gz'):
                sudo('wget {0} -O hadoop.tar.gz'.format(
                    bigdata_conf.hadoop_download_url
                ))

            sudo('rm -rf hadoop-*')
            sudo('tar -zxf hadoop.tar.gz')
            sudo('rm -rf {0}'.format(bigdata_conf.hadoop_home))
            sudo('mv hadoop-* {0}'.format(bigdata_conf.hadoop_home))