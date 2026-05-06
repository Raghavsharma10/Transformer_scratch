def parse_options(cls, options):
        """Parse plug-in specific options."""
        cls.known_modules = {
            project2module(k): v.split(",")
            for k, v in [
                x.split(":[")
                for x in re.split(r"],?", options.known_modules)[:-1]
            ]
        }