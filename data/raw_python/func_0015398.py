def package_info(self):
        """
        :return: list of package info on installed packages
        """
        import subprocess
        # create a commandline like  pip show Pillow show
        package_names = self.installed_packages()
        if not package_names:
            # No installed packages yet, so nothign to do here...
            return []

        cmdline = [sys.executable, "-mpip"]
        for name in package_names:
            cmdline.extend(["show", name])

        output = subprocess.check_output(cmdline)
        # Python 3 fix
        if not isinstance(output, str):
            # Some package info is encoded in Latin-1 or something other than
            # UTF8.  Replace non-UTF characters with '?' instead of crashing.
            output = str(output, encoding='UTF-8', errors='replace')

        # parse output that looks like this example
        """
        ---
        Name: Pillow
        Version: 2.8.1
        Location: /mnt/data/home/stu/.virtualenvs/shoebot-setup/lib/python2.7/site-packages/Pillow-2.8.1-py2.7-linux-x86_64.egg
        Requires:
        ---
        Name: vext.gi
        Version: 0.5.6.25
        Location: /mnt/data/home/stu/.virtualenvs/shoebot-setup/lib/python2.7/site-packages/vext.gi-0.5.6.25-py2.7.egg
        Requires: vext

        """
        results = []
        for info in output[3:].split("---"):
            d = {}
            for line in info[1:].splitlines():
                arg, _, value = line.partition(': ')
                arg = arg.lower()
                if arg == 'requires':
                    value = value.split(', ')
                d[arg] = value
            results.append(d)
        return results