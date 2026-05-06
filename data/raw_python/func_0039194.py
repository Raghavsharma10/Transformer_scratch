def svm_version_path(version):
        """
        Path to specified spark version. Accepts semantic version numbering.
        :param version: Spark version as String
        :return: String.
        """
        return os.path.join(Spark.HOME_DIR, Spark.SVM_DIR, 'v{}'.format(version))