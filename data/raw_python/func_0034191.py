def get_ids_by_program(self, program):
		"""
		Return a set containing the process IDs from rows whose
		program string equals the given program.
		"""
		return set(row.process_id for row in self if row.program == program)