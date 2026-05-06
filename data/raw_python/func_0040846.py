def spark_install(self):
        """
        download and install spark
        :return:
        """
        sudo('apt-get -y install build-essential python-dev python-six \
             python-virtualenv libcurl4-nss-dev libsasl2-dev libsasl2-modules \
             maven libapr1-dev libsvn-dev zlib1g-dev')

        with cd('/tmp'):
            if not exists('spark.tgz'):
                sudo('wget {0} -O spark.tgz'.format(
                    bigdata_conf.spark_download_url
                ))

            sudo('rm -rf spark-*')
            sudo('tar -zxf spark.tgz')
            sudo('rm -rf {0}'.format(bigdata_conf.spark_home))
            sudo('mv spark-* {0}'.format(bigdata_conf.spark_home))