def patch_egg_info(force_hg_version=False):
	"""
	A hack to replace egg_info.tagged_version with a wrapped version
	that will use the mercurial version if indicated.

	`force_hg_version` is used for hgtools itself.
	"""
	from setuptools.command.egg_info import egg_info
	from pkg_resources import safe_version
	import functools
	orig_ver = egg_info.tagged_version

	@functools.wraps(orig_ver)
	def tagged_version(self):
		vcs_param = (
			getattr(self.distribution, 'use_vcs_version', False)
			or getattr(self.distribution, 'use_hg_version', False)
		)
		using_hg_version = force_hg_version or vcs_param
		if force_hg_version:
			# disable patched `tagged_version` to avoid affecting
			#  subsequent installs in the same interpreter instance.
			egg_info.tagged_version = orig_ver
		if using_hg_version:
			result = safe_version(self.distribution.get_version())
		else:
			result = orig_ver(self)
		self.tag_build = result
		return result
	egg_info.tagged_version = tagged_version