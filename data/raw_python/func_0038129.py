def validate(self, read_tuple_name):
        """Check RNF validity of a read tuple.

		Args:
			read_tuple_name (str): Read tuple name to be checked.s
		"""
        if reg_lrn.match(read_tuple_name) is None:
            self.report_error(
                read_tuple_name=read_tuple_name,
                error_name="wrong_read_tuple_name_structure",
                message="'{}' is not matched".format(reg_lrn),
            )
        else:
            parts = read_tuple_name.split("__")

            if reg_prefix_part.match(parts[0]) is None:
                self.report_error(
                    read_tuple_name=read_tuple_name,
                    error_name="wrong_prefix_part",
                    message="'{}' is not matched".format(reg_prefix_part),
                )

            if reg_id_part.match(parts[1]) is None:
                self.report_error(
                    read_tuple_name=read_tuple_name,
                    error_name="wrong_id_part",
                    message="'{}' is not matched".format(reg_id_part),
                )

            if reg_segmental_part.match(parts[2]) is None:
                self.report_error(
                    read_tuple_name=read_tuple_name,
                    error_name="wrong_segmental_part",
                    message="'{}' is not matched".format(reg_segmental_part),
                )

            if reg_suffix_part.match(parts[3]) is None:
                self.report_error(
                    read_tuple_name=read_tuple_name,
                    error_name="wrong_suffix_part",
                    message="'{}' is not matched".format(reg_suffix_part),
                )

            if not self.rnf_profile.check(read_tuple_name):
                self.report_error(
                    read_tuple_name=read_tuple_name,
                    error_name="wrong_profile",
                    message="Read has a wrong profile (wrong widths). It should be: {} but it is: {}.".format(
                        self.rnf_profile,
                        rnftools.rnfformat.RnfProfile(read_tuple_name=read_tuple_name),
                    ),
                    warning=True,
                )