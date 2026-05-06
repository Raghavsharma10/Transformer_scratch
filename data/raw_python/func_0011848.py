def add_config(self, config):
        """
        Update internel configuration dict with config and recheck
        """
        for attr in self.__fixed_attrs:
            if attr in config:
                raise Exception("cannot set '%s' outside of init", attr)

        # pre checkout

        stages = config.get('stages', None)
        if stages:
            self.stages = stages

        # maybe pre checkout

        # validate options
        self.__dry_run = config.get('dry_run', False)

        self.system = str.lower(platform.system())

        self.__start = config.get('start', None)
        self.__end = config.get('end', None)
        self.__only = config.get('only', None)

        self.__build_docs = config.get('build_docs', False)
        self.__chatty = config.get('chatty', False)
        self.__clean = config.get('clean', False)
        self.__devel = config.get('devel', False)
        self.__debug = config.get('debug', False)
        self.__skip_libcheck = config.get('skip_libcheck', False)
        self.__debuginfo = config.get('debuginfo', False)
        self.__release = config.get('release', False)
        self.__skip_unit = config.get('skip_unit', False)
        self.__static = config.get('static', False)
        self.__make_dash_j = int(config.get('j', 0))
        self.__target_only = config.get('target_only', None)

        bits = config.get('bits', None)
        if bits:
            self.bits = int(bits)
        else:
            self.bits = self.sys_bits

        self.compiler = config.get('compiler', None)
        self.test_config = config.get('test_config', '-')
        if not self.test_config:
            self.test_config = '-'
        self.use_ccache = config.get('use_ccache', False)
        self.tmpl_engine = config.get('tmpl_engine', 'jinja2')
        self.__write_codec = config.get('write_codec', None)
        self.__codec = None

        # TODO move out of init
        if not config.get('skip_env_check', False):
            if "LD_LIBRARY_PATH" in os.environ:
                raise Exception("environment variable LD_LIBRARY_PATH is set")

        self.check_config()