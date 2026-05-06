def report_error(self, read_tuple_name, error_name, wrong="", message="", warning=False):
        """Report an error.

		Args:
			read_tuple_name (): Name of the read tuple.
			error_name (): Name of the error.
			wrong (str): What is wrong. 
			message (str): Additional msessage to be printed.
			warning (bool): Warning (not an error).
		"""
        if (not self.report_only_first) or (error_name not in self.reported_errors):
            print("\t".join(["error" if warning == False else "warning", read_tuple_name, error_name, wrong, message]))
        self.reported_errors.add(error_name)
        if warning:
            self.warning_has_been_reported = True
        else:
            self.error_has_been_reported = True