def mac_app_exists(app):
	'''Check if 'app' is installed (OS X).

	Check if the given applications is installed on this OS X system.

	Args:
			app (str): The application name.

	Returns:
			bool: Is the app installed or not?
	'''

	APP_CHECK_APPLESCRIPT = '''try
	tell application "Finder"
		set appname to name of application file id "%s"
		return 0
	end tell
	on error err_msg number err_num
		return 1
	end try'''

	with open('/tmp/app_check.AppleScript', 'w') as f:
		f.write(APP_CHECK_APPLESCRIPT % app)

	app_check_proc = sp.Popen(
		['osascript', '-e', '/tmp/app_check.AppleScript'])

	if app_check_proc.wait() != 0:
		return False

	else:
		return True