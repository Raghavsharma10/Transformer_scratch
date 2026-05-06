def clean_virtualenv(self):
        """
        Empty our virtualenv so that new (or older) dependencies may be
        installed
        """
        self.user_run_script(
            script=scripts.get_script_path('clean_virtualenv.sh'),
            args=[],
            rw_venv=True,
            )