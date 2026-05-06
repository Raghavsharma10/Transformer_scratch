def installed_packages(self):
        """ :return: list of installed packages """
        packages = []
        CMDLINE = [sys.executable, "-mpip", "freeze"]
        try:
            for package in subprocess.check_output(CMDLINE) \
                    .decode('utf-8'). \
                    splitlines():
                for comparator in ["==", ">=", "<=", "<", ">"]:
                    if comparator in package:
                        # installed package names usually look like Pillow==2.8.1
                        # ignore others, like external packages that pip show
                        # won't understand
                        name = package.partition(comparator)[0]
                        packages.append(name)
        except RuntimeError as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Exception checking existing packages.")
                logger.debug("cmdline: %s", CMDLINE)
                ex_type, ex, tb = sys.exc_info()
                traceback.print_tb(tb)
                logger.debug()
        return packages