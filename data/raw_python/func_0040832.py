def update_source_list(self):
        """
        update ubuntu 16 source list
        :return: 
        """
        with cd('/etc/apt'):
            sudo('mv sources.list sources.list.bak')
            put(StringIO(bigdata_conf.ubuntu_source_list_16),
                'sources.list', use_sudo=True)
            sudo('apt-get update -y --fix-missing')