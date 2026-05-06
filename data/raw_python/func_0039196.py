def rename_unzipped_folder(version):
        """
        Renames unzipped spark version folder to the release tag.
        :param version: version from release tag.
        :return:
        """

        for filename in os.listdir(Spark.svm_path()):
            if fnmatch.fnmatch(filename, 'apache-spark-*'):
                return os.rename(os.path.join(Spark.svm_path(), filename), Spark.svm_version_path(version))

        raise SparkInstallationError("Unable to find unzipped Spark folder in {}".format(Spark.svm_path()))