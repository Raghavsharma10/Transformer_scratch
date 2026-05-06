def run_json(self):
        """
        Run checks on self.files, printing json object
        containing information relavent to the CS50 IDE plugin at the end.
        """
        checks = {}
        for file in self.files:
            try:
                results = self._check(file)
            except Error as e:
                checks[file] = {
                    "error": e.msg
                }
            else:
                checks[file] = {
                    "score": results.score,
                    "comments": results.comment_ratio >= results.COMMENT_MIN,
                    "diff": "<pre>{}</pre>".format("\n".join(self.html_diff(results.original, results.styled))),
                }

        json.dump(checks, sys.stdout, indent=4)
        print()