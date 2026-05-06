def get_version():
        """Returns the Postgres version in tuple form, e.g: (9, 1)"""
        cmd = [PostgresFinder.find_root() / 'pg_ctl', '--version']
        results = subprocess.check_output(cmd).decode('utf-8')
        match = re.search(r'(\d+\.\d+(\.\d+)?)', results)
        if match:
            ver_string = match.group(0)
            return tuple(int(x) for x in ver_string.split('.'))