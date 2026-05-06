def set_wallpaper(image):
	'''Set the desktop wallpaper.

	Sets the desktop wallpaper to an image.

	Args:
			image (str): The path to the image to be set as wallpaper.
	'''

	desktop_env = system.get_name()

	if desktop_env in ['gnome', 'unity', 'cinnamon', 'pantheon', 'mate']:
		uri = 'file://%s' % image

		SCHEMA = 'org.gnome.desktop.background'
		KEY = 'picture-uri'

		if desktop_env == 'mate':
			uri = image

			SCHEMA = 'org.mate.background'
			KEY = 'picture-filename'

		try:
			from gi.repository import Gio

			gsettings = Gio.Settings.new(SCHEMA)
			gsettings.set_string(KEY, uri)
		except ImportError:
			try:
				gsettings_proc = sp.Popen(
					['gsettings', 'set', SCHEMA, KEY, uri])
			except:  # MATE < 1.6
				sp.Popen(['mateconftool-2',
						  '-t',
						  'string',
						  '--set',
						  '/desktop/mate/background/picture_filename',
						  '%s' % image],
						 stdout=sp.PIPE)
			finally:
				gsettings_proc.communicate()

				if gsettings_proc.returncode != 0:
					sp.Popen(['mateconftool-2',
							  '-t',
							  'string',
							  '--set',
							  '/desktop/mate/background/picture_filename',
							  '%s' % image])

	elif desktop_env == 'gnome2':
		sp.Popen(
			['gconftool-2',
			 '-t',
			 'string',
			 '--set',
			 '/desktop/gnome/background/picture_filename',
			 image]
		)

	elif desktop_env == 'kde':
		# This probably only works in Plasma 5+

		kde_script = dedent(
		'''\
		var Desktops = desktops();
		for (i=0;i<Desktops.length;i++) {{
			d = Desktops[i];
			d.wallpaperPlugin = "org.kde.image";
			d.currentConfigGroup = Array("Wallpaper",
										"org.kde.image",
										"General");
			d.writeConfig("Image", "file://{}")
		}}
		''').format(image)

		sp.Popen(
				['dbus-send',
				 '--session',
				 '--dest=org.kde.plasmashell',
				 '--type=method_call',
				 '/PlasmaShell',
				 'org.kde.PlasmaShell.evaluateScript',
				 'string:{}'.format(kde_script)]
		)

	elif desktop_env in ['kde3', 'trinity']:
		args = 'dcop kdesktop KBackgroundIface setWallpaper 0 "%s" 6' % image
		sp.Popen(args, shell=True)

	elif desktop_env == 'xfce4':
		# XFCE4's image property is not image-path but last-image (What?)

		list_of_properties = system.get_cmd_out(
				['xfconf-query',
				 '-R',
				 '-l',
				 '-c',
				 'xfce4-desktop',
				 '-p',
				 '/backdrop']
		)

		for i in list_of_properties.split('\n'):
			if i.endswith('last-image'):
				# The property given is a background property
				sp.Popen(
					['xfconf-query -c xfce4-desktop -p %s -s "%s"' %
						(i, image)],
					shell=True)

				sp.Popen(['xfdesktop --reload'], shell=True)

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
				desktop_conf.set('razor', config_option, image)
				with codecs.open(desktop_conf_file, 'w', encoding='utf-8', errors='replace') as f:
					desktop_conf.write(f)
		except:
			pass

	elif desktop_env in ['fluxbox', 'jwm', 'openbox', 'afterstep', 'i3']:
		try:
			args = ['feh', '--bg-scale', image]
			sp.Popen(args)
		except:
			sys.stderr.write('Error: Failed to set wallpaper with feh!')
			sys.stderr.write('Please make sre that You have feh installed.')

	elif desktop_env == 'icewm':
		args = ['icewmbg', image]
		sp.Popen(args)

	elif desktop_env == 'blackbox':
		args = ['bsetbg', '-full', image]
		sp.Popen(args)

	elif desktop_env == 'lxde':
		args = 'pcmanfm --set-wallpaper %s --wallpaper-mode=scaled' % image
		sp.Popen(args, shell=True)

	elif desktop_env == 'lxqt':
		args = 'pcmanfm-qt --set-wallpaper %s --wallpaper-mode=scaled' % image
		sp.Popen(args, shell=True)

	elif desktop_env == 'windowmaker':
		args = 'wmsetbg -s -u %s' % image
		sp.Popen(args, shell=True)

	elif desktop_env == 'enlightenment':
		args = 'enlightenment_remote -desktop-bg-add 0 0 0 0 %s' % image
		sp.Popen(args, shell=True)

	elif desktop_env == 'awesome':
		with sp.Popen("awesome-client", stdin=sp.PIPE) as awesome_client:
			command = ('local gears = require("gears"); for s = 1,'
						' screen.count() do gears.wallpaper.maximized'
						'("%s", s, true); end;') % image
			awesome_client.communicate(input=bytes(command, 'UTF-8'))

	elif desktop_env == 'windows':
		WINDOWS_SCRIPT = dedent('''
			reg add "HKEY_CURRENT_USER\Control Panel\Desktop" \
			/v Wallpaper /t REG_SZ /d  %s /f

			rundll32.exe user32.dll,UpdatePerUserSystemParameters
			''') % image

		windows_script_file = os.path.join(
			tempfile.gettempdir(), 'wallscript.bat')

		with open(windows_script_file, 'w') as f:
			f.write(WINDOWS_SCRIPT)

		sp.Popen([windows_script_file], shell=True)

		# Sometimes the method above works
		# and sometimes the one below

		SPI_SETDESKWALLPAPER = 20
		ctypes.windll.user32.SystemParametersInfoA(
			SPI_SETDESKWALLPAPER, 0, image, 0)

	elif desktop_env == 'mac':
		try:
			from appscript import app, mactypes
			app('Finder').desktop_picture.set(mactypes.File(image))
		except ImportError:
			OSX_SCRIPT = dedent(
				'''tell application "System Events"
					   set desktopCount to count of desktops
							 repeat with desktopNumber from 1 to desktopCount
							   tell desktop desktopNumber
								 set picture to POSIX file "%s"
							   end tell
							 end repeat
				 end tell''') % image

			sp.Popen(['osascript', OSX_SCRIPT])
	else:
		try:
			sp.Popen(['feh', '--bg-scale', image])
			# feh is nearly a catch-all for Linux WMs
		except:
			pass