def elastic_install(self):
        """
        elasticsearch install
        :return:
        """
        with cd('/tmp'):
            if not exists('elastic.deb'):
                sudo('wget {0} -O elastic.deb'.format(
                    bigdata_conf.elastic_download_url
                ))

            sudo('dpkg -i elastic.deb')
            sudo('apt-get install -y')