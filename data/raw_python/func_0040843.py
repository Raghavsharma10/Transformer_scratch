def kibana_install(self):
        """
        kibana install
        :return:
        """
        with cd('/tmp'):
            if not exists('kibana.deb'):
                sudo('wget {0} -O kibana.deb'.format(
                    bigdata_conf.kibana_download_url
                ))

            sudo('dpkg -i kibana.deb')
            sudo('apt-get install -y')