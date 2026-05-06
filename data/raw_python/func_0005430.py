def search_project_root():
        """
        Search your Django project root.

        returns:
            - path:string  Django project root path
        """

        while True:

            current = os.getcwd()

            if pathlib.Path("Miragefile.py").is_file() or pathlib.Path("Miragefile").is_file():
                return current
            elif os.getcwd() == "/":
                raise FileNotFoundError
            else:
                os.chdir("../")