def _check_dir(self, dirname):
        """ Check if dir exists, if not: give warning and die"""
        if not os.path.exists(dirname):
            print("Directory %s does not exist!" % dirname)
            sys.exit(1)