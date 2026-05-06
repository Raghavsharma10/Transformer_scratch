def write_results(self, data, name=None):
        """
        Write JSON to file with the specified name.

        :param name: Path to the file to be written to. If no path is passed
                     a new JSON file "results.json" will be created in the
                     current working directory.
        :param output: JSON object.
        """

        if name:
            filepath = os.path.abspath(name)
        else:
            filepath = os.path.join(os.path.getcwd(), "results.json")

        with open(filepath, "w", encoding="utf8") as f:
            try:
                f.write(unicode(json.dumps(data, indent=4)))
            except NameError:
                f.write(json.dumps(data, indent=4))