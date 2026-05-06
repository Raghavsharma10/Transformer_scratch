def brew(clean=False):
	"""
	Handle homebrew on macOS
	"""
	# see if homebrew is installed
	try: cmd('which brew')
	except:
		return

	print('-[brew]----------')
	cmd('brew update')
	p = cmd('brew outdated')
	if not p: return
	pkgs = getPackages(p)
	for p in pkgs:
		cmd('brew upgrade {}'.format(p), run=global_run)

	if clean:
		print(' > brew prune old sym links and cleanup')
		cmd('brew prune')
		cmd('brew cleanup')