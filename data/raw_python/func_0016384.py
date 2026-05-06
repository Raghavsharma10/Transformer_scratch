def run(self):
        """runner"""

        # change version in code and changelog before running this
        subprocess.check_call("git commit CHANGELOG.rst pyros/_version.py CHANGELOG.rst -m 'v{0}'".format(__version__), shell=True)
        subprocess.check_call("git push", shell=True)

        print("You should verify travis checks, and you can publish this release with :")
        print("  python setup.py publish")
        sys.exit()