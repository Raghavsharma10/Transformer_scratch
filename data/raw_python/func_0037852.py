def infer_next_version(last_version, increment):
		"""
		Given a simple application version (as a StrictVersion),
		and an increment (1.0, 0.1, or 0.0.1), guess the next version.

		Set up a shorthand for examples

		>>> def VM_infer(*params):
		...     return str(VersionManagement.infer_next_version(*params))

		>>> VM_infer('3.2', '0.0.1')
		'3.2.1'
		>>> VM_infer(StrictVersion('3.2'), '0.0.1')
		'3.2.1'
		>>> VM_infer('3.2.3', '0.1')
		'3.3'
		>>> VM_infer('3.1.2', '1.0')
		'4.0'

		Subversions never increment parent versions

		>>> VM_infer('3.0.9', '0.0.1')
		'3.0.10'

		If it's a prerelease version, just remove the prerelease.

		>>> VM_infer('3.1a1', '0.0.1')
		'3.1'

		If there is no last version, use the increment itself

		>>> VM_infer(None, '0.1')
		'0.1'
		"""
		if last_version is None:
			return increment
		last_version = SummableVersion(str(last_version))
		if last_version.prerelease:
			last_version.prerelease = None
			return str(last_version)
		increment = SummableVersion(increment)
		sum = last_version + increment
		sum.reset_less_significant(increment)
		return sum