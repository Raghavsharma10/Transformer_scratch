def version_calc(dist, attr, value):
	"""
	Handler for parameter to setup(use_vcs_version=value)
	attr should be 'use_vcs_version' (also allows use_hg_version for
		compatibility).
	bool(value) should be true to invoke this plugin.
	value may optionally be a dict and supply options to the plugin.
	"""
	expected_attrs = 'use_hg_version', 'use_vcs_version'
	if not value or attr not in expected_attrs:
		return
	options = value if isinstance(value, dict) else {}
	dist.metadata.version = calculate_version(options)
	patch_egg_info()