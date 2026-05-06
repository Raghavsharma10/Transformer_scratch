def exit(self) -> None:
        """Raise SystemExit with correct status code and output logs."""
        total = sum(len(logs) for logs in self.logs.values())
        if self.json:
            self.logs['total'] = total
            print(json.dumps(self.logs, indent=self.indent))

        else:
            for name, log in self.logs.items():
                if not log or self.parser[name].as_bool("quiet"):
                    continue

                print("[[{0}]]".format(name))
                getattr(snekchek.format, name + "_format")(log)
                print("\n")

            print("-" * 30)
            print("Total:", total)

        sys.exit(self.status_code)