def main(self) -> None:
        """The main function for generating the config file"""
        path = ask_path("where should the config be stored?", ".snekrc")

        conf = configobj.ConfigObj()

        tools = self.get_tools()
        for tool in tools:
            conf[tool] = getattr(self, tool)()  # pylint: disable=assignment-from-no-return
        conf.filename = path
        conf.write()

        print("Written config file!")

        if "pylint" in tools:
            print(
                "Please also run `pylint --generate-rcfile` to complete setup")