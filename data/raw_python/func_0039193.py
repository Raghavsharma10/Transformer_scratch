def download_source(version):
        """
        Download Spark version. Uses same name as release tag without the leading 'v'.
        :param version: Version number to download.
        :return: None
        """
        local_filename = 'v{}.zip'.format(Spark.svm_version_path(version))
        Spark.download(Spark.spark_versions()['v{}'.format(version)], local_filename)