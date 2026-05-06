def pip(usr_pswd=None):
	"""
	This updates one package at a time.

	Could do all at once:
		pip list --outdated | cut -d' ' -f1 | xargs pip install --upgrade
	"""
	# see if pip is installed
	try: cmd('which pip')
	except:
		return

	print('-[pip]----------')
	p = cmd('pip list --outdated')
	if not p: return
	pkgs = getPackages(p)

	# update pip and setuptools first
	for i, p in enumerate(pkgs):
		if p in ['pip', 'setuptools']:
			cmd('pip install -U ' + p, usr_pwd=usr_pswd, run=global_run)
			pkgs.pop(i)

	# update the rest of them
	for p in pkgs:
		cmd('pip install -U ' + p, usr_pwd=usr_pswd, run=global_run)