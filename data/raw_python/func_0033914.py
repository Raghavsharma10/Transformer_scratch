def set_artefact_path(self, path_to_zip_file):
        """
        Set the route to the local file to deploy
        :param path_to_zip_file: 
        :return: 
        """
        self.config["deploy"]["deploy_file"] = path_to_zip_file
        return {'ZipFile': self.build.read(self.config["deploy"]["deploy_file"])}