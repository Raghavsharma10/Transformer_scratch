def get_wallpaper():
	'''Get the desktop wallpaper.

	Get the current desktop wallpaper.

	Returns:
			str: The path to the current wallpaper.
	'''

	desktop_env = system.get_name()

	if desktop_env in ['gnome', 'unity', 'cinnamon', 'pantheon', 'mate']:
		SCHEMA = 'org.gnome.desktop.background'
		KEY = 'picture-uri'

		if desktop_env == 'mate':
			SCHEMA = 'org.mate.background'
			KEY = 'picture-filename'

		try:
			from gi.repository import Gio

			gsettings = Gio.Settings.new(SCHEMA)
			return gsettings.get_string(KEY).replace('file://', '')
		except ImportError:
			try:
				return system.get_cmd_out(
					['gsettings', 'get', SCHEMA, KEY]).replace('file://', '')
			except:  # MATE < 1.6
				return system.get_cmd_out(
						['mateconftool-2', '-t', 'string', '--get',
						 '/desktop/mate/background/picture_filename']
						).replace('file://', '')

	elif desktop_env == 'gnome2':
		args = ['gconftool-2', '-t', 'string', '--get',
				'/desktop/gnome/background/picture_filename']
		return system.get_cmd_out(args).replace('file://', '')

	elif desktop_env == 'kde':
		conf_file = directories.get_config_file(
				'plasma-org.kde.plasma.desktop-appletsrc')[0]
		with open(conf_file) as f:
			contents = f.read()

		contents = contents.splitlines()

		contents = contents[
			contents.index(
			'[Containments][8][Wallpaper][org.kde.image][General]') +
			1].split(
			'=',
			1
			)

		return contents[len(contents) - 1].strip().replace('file://', '')

	elif desktop_env == 'xfce4':
		# XFCE4's image property is not image-path but last-image (What?)

		list_of_properties = system.get_cmd_out(
			['xfconf-query', '-R', '-l', '-c', 'xfce4-desktop', '-p',
			 '/backdrop'])

		for i in list_of_properties.split('\n'):
			if i.endswith('last-image') and 'workspace' in i:
				# The property given is a background property
				return system.get_cmd_out(
					['xfconf-query', '-c', 'xfce4-desktop', '-p', i])

	elif desktop_env == 'razor-qt':
		desktop_conf = configparser.ConfigParser()
		# Development version

		desktop_conf_file = os.path.join(
			get_config_dir('razor')[0], 'desktop.conf')

		if os.path.isfile(desktop_conf_file):
			config_option = r'screens\1\desktops\1\wallpaper'

		else:
			desktop_conf_file = os.path.join(
				os.path.expanduser('~'), '.razor/desktop.conf')
			config_option = r'desktops\1\wallpaper'

		desktop_conf.read(os.path.join(desktop_conf_file))
		try:
			if desktop_conf.has_option('razor', config_option):
				return desktop_conf.get('razor', config_option)
		except:
			pass

	elif desktop_env in ['fluxbox', 'jwm', 'openbox', 'afterstep', 'i3']:
		# feh stores last feh command in '~/.fehbg'
		# parse it
		with open(os.path.expanduser('~/.fehbg')) as f:
			fehbg = f.read()

		fehbg = fehbg.split('\n')

		for line in fehbg:
			if '#!' in line:
				fehbg.remove(line)

		fehbg = fehbg[0]

		for i in fehbg.split(' '):
			if not i.startswith("-"):
				if not i.startswith('feh'):
					if not i in ['', ' ', '  ', '\n']:
						return(i.replace("'", ''))

	# TODO: way to get wallpaper for desktops which are commented-out below
	elif desktop_env == 'icewm':
		with open(os.path.expanduser('~/.icewm/preferences')) as f:
			for line in f:
				if line.startswith('DesktopBackgroundImage'):
					return os.path.expanduser(line.strip().split(
						'=', 1)[1].strip().replace('"', '').replace("'", ''))

	elif desktop_env == 'awesome':
		conf_file = os.path.join(
				directories.get_config_dir('awesome')[0],
				'rc.lua')

		with open(conf_file) as f:
			for line in f:
				if line.startswith('theme_path'):
					awesome_theme = line.strip().split('=', 1)
					awesome_theme = awesome_theme[
						len(awesome_theme) -
						1].strip().replace(
						'"',
						'').replace(
						"'",
						'')

		with open(os.path.expanduser(awesome_theme)) as f:
			for line in f:
				if line.startswith('theme.wallpaper'):
					awesome_wallpaper = line.strip().split('=', 1)
					awesome_wallpaper = awesome_wallpaper[
						len(awesome_wallpaper) -
						1].strip().replace(
						'"',
						'').replace(
						"'",
						'')

					return os.path.expanduser(awesome_wallpaper)

	# elif desktop_env == 'blackbox':
	# 	args = ['bsetbg', '-full', image]
	# 	sp.Popen(args)
	#
	# elif desktop_env == 'lxde':
	# 	args = 'pcmanfm --set-wallpaper %s --wallpaper-mode=scaled' % image
	# 	sp.Popen(args, shell=True)
	#
	# elif desktop_env == 'lxqt':
	# 	args = 'pcmanfm-qt --set-wallpaper %s --wallpaper-mode=scaled' % image
	# 	sp.Popen(args, shell=True)
	#
	# elif desktop_env == 'windowmaker':
	# 	args = 'wmsetbg -s -u %s' % image
	# 	sp.Popen(args, shell=True)
	#
	# elif desktop_env == 'enlightenment':
	#	args = 'enlightenment_remote -desktop-bg-add 0 0 0 0 %s' % image
	#	sp.Popen(args, shell=True)
	#
	# elif desktop_env == 'awesome':
	# 	with sp.Popen("awesome-client", stdin=sp.PIPE) as awesome_client:
	# 		command = 'local gears = require("gears"); for s = 1, screen.count()
	#       do gears.wallpaper.maximized("%s", s, true); end;' % image
	# 		awesome_client.communicate(input=bytes(command, 'UTF-8'))

	elif desktop_env == 'windows':
		WINDOWS_SCRIPT = ('reg query "HKEY_CURRENT_USER\Control'
						  ' Panel\Desktop\Desktop"')

		return system.get_cmd_out(WINDOWS_SCRIPT)

	elif desktop_env == 'mac':
		try:
			from appscript import app
			app('Finder').desktop_picture.get()
		except ImportError:
			OSX_SCRIPT = ('tell app "finder" to get posix path'
						  ' of (get desktop picture as alias)')

			return system.get_cmd_out(['osascript', OSX_SCRIPT])