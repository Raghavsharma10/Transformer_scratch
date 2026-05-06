def build_from_source(version, **kwargs):
        """
        Builds specified Spark version from source.
        :param version:
        :param kwargs:
        :return: (Integer) Status code of build/mvn command.
        """
        mvn = os.path.join(Spark.svm_version_path(version), 'build', 'mvn')
        Spark.chmod_add_excute(mvn)
        p = subprocess.Popen([mvn, '-DskipTests', 'clean', 'package'], cwd=Spark.svm_version_path(version))
        p.wait()
        return p.returncode