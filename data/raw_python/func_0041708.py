def npm(usr_pwd=None, clean=False):
	"""
	Handle npm for Node.js
	"""
	# see if node is installed
	try: cmd('which npm')
	except:
		return

	print('-[npm]----------')
	# awk, ignore 1st line and grab 1st word
	p = cmd("npm outdated -g | awk 'NR>1 {print $1}'")
	if not p: return
	pkgs = getPackages(p)

	for p in pkgs:
		cmd('{} {}'.format('npm update -g ', p), usr_pwd=usr_pwd, run=global_run)