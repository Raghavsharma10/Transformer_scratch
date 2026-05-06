def update_history(self) -> None:
        """
        Update messaging history on disk.

        :returns: None
        """
        self.log.debug(f"Saving history. History is: \n{self.history}")

        jsons = []
        for item in self.history:
            json_item = item.__dict__

            # Convert sub-entries into JSON as well.
            json_item["output_records"] = self._parse_output_records(item)

            jsons.append(json_item)

        if not path.isfile(self.history_filename):
            open(self.history_filename, "a+").close()

        with open(self.history_filename, "w") as f:
            json.dump(jsons, f, default=lambda x: x.__dict__.copy(), sort_keys=True, indent=4)
            f.write("\n")