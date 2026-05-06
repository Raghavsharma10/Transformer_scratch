def _write_properties_file(self):
        """Write an RDP training properties file manually.
        """
        # The properties file specifies the names of the files in the
        # training directory.  We use the example properties file
        # directly from the rdp_classifier distribution, which lists
        # the default set of files created by the application.  We
        # must write this file manually after generating the
        # training data.
        properties_fp = os.path.join(self.ModelDir, self.PropertiesFile)
        properties_file = open(properties_fp, 'w')
        properties_file.write(
            "# Sample ResourceBundle properties file\n"
            "bergeyTree=bergeyTrainingTree.xml\n"
            "probabilityList=genus_wordConditionalProbList.txt\n"
            "probabilityIndex=wordConditionalProbIndexArr.txt\n"
            "wordPrior=logWordPrior.txt\n"
            "classifierVersion=Naive Bayesian rRNA Classifier Version 1.0, "
            "November 2003\n"
        )
        properties_file.close()