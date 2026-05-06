def search_app_root():
        """
        Search your Django application root

        returns:
            - (String) Django application root path
        """
        while True:

            current = os.getcwd()

            if pathlib.Path("apps.py").is_file():
                return current
            elif pathlib.Path.cwd() == "/":
                raise FileNotFoundError
            else:
                os.chdir("../")