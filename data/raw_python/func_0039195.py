def unzip(filename):
        """
        Unzips specified file into ~/.svm
        :param filename:
        :return:
        """
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(Spark.svm_path())
        return True