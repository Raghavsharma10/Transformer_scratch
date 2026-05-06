def user_agent(self):
        """
        its a user agent string!
        """
        version = ""
        project_root = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(project_root, 'VERSION')) as version_file:
            version = version_file.read().strip()

        return "Python Snow Api Client (Version %s)" % version