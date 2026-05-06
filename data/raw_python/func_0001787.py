def get_tools(self) -> list:
        """Lets the user enter the tools he want to use"""
        tools = "flake8,pylint,vulture,pyroma,isort,yapf,safety,dodgy,pytest,pypi".split(
            ",")
        print("Available tools: {0}".format(",".join(tools)))
        answer = ask_list("What tools would you like to use?",
                          ["flake8", "pytest"])

        if any(tool not in tools for tool in answer):
            print("Invalid answer, retry.")
            self.get_tools()
        return answer