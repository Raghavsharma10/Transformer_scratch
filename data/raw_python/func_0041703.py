def cmd(str, print_ret=False, usr_pwd=None, run=True):
	"""
	Executes a command and throws an exception on error.
	in:
		str - command
		print_ret - print command return
		usr_pwd - execute command as another user (user_name, password)
		run - really execute command?
	out:
		returns the command output
	"""
	if usr_pwd:
		str = 'echo {} | sudo -u {} {} '.format(usr_pwd[1], usr_pwd[0], str)

	print('  [>] {}'.format(str))

	if run:
		err, ret = commands.getstatusoutput(str)
	else:
		err = None
		ret = None

	if err:
		print('  [x] {}'.format(ret))
		raise Exception(ret)
	if ret and print_ret:
		lines = ret.split('\n')
		for line in lines:
			print('  [<] {}'.format(line))
	return ret