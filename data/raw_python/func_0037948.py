def is_running(process):
	'''
	Check if process is running.

	Check if the given process name is running or not.

	Note:
			On a Linux system, kernel threads (like	``kthreadd`` etc.)
			are excluded.

	Args:
			process (str): The name of the process.

	Returns:
			bool: Is the process running?
	'''

	if os.name == 'nt':
		process_list = get_cmd_out(['tasklist', '/v'])
		return process in process_list

	else:
		process_list = get_cmd_out('ps axw | awk \'{print $5}\'')

		for i in process_list.split('\n'):
			# 'COMMAND' is the column heading
			# [*] indicates kernel-level processes like \
			# kthreadd, which manages threads in the Linux kernel
			if not i == 'COMMAND' or i.startswith('['):
				if i == process:
					return True

				elif os.path.basename(i) == process:
					# check i without executable path
					# for example, if 'process' arguments is 'sshd'
					# and '/usr/bin/sshd' is listed in ps, return True
					return True

	return False