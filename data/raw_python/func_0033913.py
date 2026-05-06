def run(self, src_project=None, path_to_zip_file=None):
        """Run deploy the lambdas defined in our project.
        Steps:
        * Build Artefact
        * Read file or deploy to S3. It's defined in config["deploy"]["deploy_method"]
        * Reload conf with deploy changes
        * check lambda if exist
            * Create Lambda
            * Update Lambda
                
        
        :param src_project: str. Name of the folder or path of the project where our code lives
        :param path_to_zip_file: str. 
        :return: bool
        """
        if path_to_zip_file:
            code = self.set_artefact_path(path_to_zip_file)
        elif not self.config["deploy"].get("deploy_file", False):
            code = self.build_artefact(src_project)
        else:
            code = self.set_artefact_path(self.config["deploy"].get("deploy_file"))

        self.set_artefact(code=code)
        # Reload conf because each lambda conf need to read again the global conf
        self.config.reload_conf()

        self.deploy()

        return True