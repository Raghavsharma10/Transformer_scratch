def install_package_requirements(self, psrc, stream_output=None):
        """
        Install from requirements.txt file found in psrc

        :param psrc: name of directory in environment directory
        """
        package = self.target + '/' + psrc
        assert isdir(package), package
        reqname = '/requirements.txt'
        if not exists(package + reqname):
            reqname = '/pip-requirements.txt'
            if not exists(package + reqname):
                return
        return self.user_run_script(
            script=scripts.get_script_path('install_reqs.sh'),
            args=['/project/' + psrc + reqname],
            rw_venv=True,
            rw_project=True,
            stream_output=stream_output
            )