def terminal(exec_='', background=False, shell_after_cmd_exec=False,
			 keep_open_after_cmd_exec=False, return_cmd=False):
	'''Start the default terminal emulator.

	Start the user's preferred terminal emulator, optionally running a command in it.

	**Order of starting**
			Windows:
					Powershell

			Mac:
					- iTerm2
					- Terminal.app

			Linux/Unix:
					- ``$TERMINAL``
					- ``x-terminal-emulator``
					- Terminator
					- Desktop environment's terminal
					- gnome-terminal
					- urxvt
					- rxvt
					- xterm

	Args:
			exec\_			   (str) : An optional command to run in the opened terminal emulator. Defaults to empty (no command).
			background		   (bool): Run the terminal in the background, instead of waiting for completion. Defaults to ``False``.
			shell_after_cmd_exec (bool): Start the user's shell after running the command (see exec_). Defaults to `False`.
			return_cmd		   (bool): Returns the command used to start the terminal (str) instead of running it. Defaults to ``False``.
	Returns:
			str: Only if ``return_cmd``, returns the command to run the terminal instead of running it. Else returns nothing.
	'''

	desktop_env = system.get_name()

	if not exec_:
		shell_after_cmd_exec = True

	if desktop_env == 'windows':
		terminal_cmd_str = 'start powershell.exe'

	if desktop_env == 'mac':
		# Try iTerm2 first, apparently most popular Mac Terminal
		if mac_app_exists('iTerm2'):
			terminal_cmd_str = 'open -a iTerm2'

		else:
			terminal_cmd_str = 'open -a Terminal'

	else:

		# sensible-terminal

		if os.getenv('TERMINAL'):
			# Not everywhere, but if user *really* has a preference, they will
			# set this

			terminal_cmd_str = os.getenv('TERMINAL')

		elif system.is_in_path('x-terminal-emulator'):
			# This is a convenience script that launches terminal based on
			# user preferences.
			# This is not available on some distros (but most have it)
			# so try this first
			terminal_cmd_str = 'x-terminal-emulator'

		elif system.is_in_path('terminator'):
			terminal_cmd_str = 'terminator'

		elif desktop_env in ['gnome', 'unity', 'cinnamon', 'gnome2']:
			terminal_cmd_str = 'gnome-terminal'

		elif desktop_env == 'xfce4':
			terminal_cmd_str = 'xfce4-terminal'

		elif desktop_env == 'kde' or desktop_env == 'trinity':
			terminal_cmd_str = 'konsole'

		elif desktop_env == 'mate':
			terminal_cmd_str = 'mate-terminal'

		elif desktop_env == 'i3':
			terminal_cmd_str = 'i3-sensible-terminal'

		elif desktop_env == 'pantheon':
			terminal_cmd_str = 'pantheon-terminal'

		elif desktop_env == 'enlightenment':
			terminal_cmd_str = 'terminology'

		elif desktop_env == 'lxde' or desktop_env == 'lxqt':
			terminal_cmd_str = 'lxterminal'

		else:
			if system.is_in_path('gnome-terminal'):
				terminal_cmd_str = 'gnome-terminal'

			elif system.is_in_path('urxvt'):
				terminal_cmd_str = 'urxvt'

			elif system.is_in_path('rxvt'):
				terminal_cmd_str = 'rxvt'

			elif system.is_in_path('xterm'):
				terminal_cmd_str = 'xterm'

	if exec_:
		if desktop_env == 'windows':
			if keep_open_after_cmd_exec and not shell_after_cmd_exec:
				exec_ += '; pause'

			if os.path.isfile(exec_):
				terminal_cmd_str += exec_

			else:
				terminal_cmd_str += ' -Command ' + '"' + exec_ + '"'

			if shell_after_cmd_exec:
				terminal_cmd_str += ' -NoExit'

		else:
			if keep_open_after_cmd_exec and not shell_after_cmd_exec:
				exec_ += '; read'

			if shell_after_cmd_exec:
				exec_ += '; ' + os.getenv('SHELL')

			if desktop_env == 'mac':
				terminal_cmd_str += ' sh -c {}'.format(shlex.quote(exec_))

			else:
				terminal_cmd_str += ' -e {}'.format(
					shlex.quote('sh -c {}'.format(shlex.quote(exec_))))

	if return_cmd:
		return terminal_cmd_str

	terminal_proc = sp.Popen([terminal_cmd_str], shell=True, stdout=sp.PIPE)

	if not background:
		# Wait for process to complete
		terminal_proc.wait()