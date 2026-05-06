def check_config(self):
        """
        called after config was modified to sanity check
        raises on error
        """

        # sanity checks - no config access past here
        if not getattr(self, 'stages', None):
            raise NotImplementedError("member variable 'stages' must be defined")
        # start at stage
        if self.__start:
            self.__stage_start = self.find_stage(self.__start)
        else:
            self.__stage_start = 0
        # end at stage
        if self.__end:
            self.__stage_end = self.find_stage(self.__end) + 1
            self.opt_end = self.__end
        else:
            self.__stage_end = len(self.stages)
        # only stage
        if self.__only:
            if self.__start or self.__end:
                raise Exception(
                    "stage option 'only' cannot be used with start or end")
            self.__stage_start = self.find_stage(self.__only)
            self.__stage_end = self.__stage_start + 1

        if self.__devel:
            self.__devel = True
            # force deploy skip
            if self.__stage_end >= len(self.stages):
                self.status_msg("removing deploy stage for development build")
# XXX                self.__stage_end = self.__stage_end - 1


        if self.stage_start >= self.stage_end:
            raise Exception("start and end produce no stages")

        if self.bits not in [32, 64]:
            raise Exception(
                "can't do a %d bit build: unknown build process" % self.bits)

        if self.bits == 64 and not self.is_64b:
            raise Exception(
                "this machine is not 64 bit, cannot perform 64 bit build")

        if self.system == 'windows':
            self.compilertag = 'vc10'
        elif self.system == 'linux':
            self.compilertag = 'gcc44'
        else:
            raise RuntimeError("can't decide compilertag on " + self.system)

        self.build_suffix = ''
        if not self.is_unixy:
            if self.__static:
                runtime = 'MT'
            else:
                runtime = 'MD'
            if self.__release:
                self.configuration_name = 'Release'
            else:
                runtime += 'd'
                self.configuration_name = 'Debug'

            self.build_suffix = '-' + runtime
            self.runtime = runtime
        else:
            self.configuration_name = 'CFNAME_INVALID_ON_LINUX'
            self.runtime = 'RUNTIME_INVALID_ON_LINUX'

        if self.test_config != '-':
            self.test_config = os.path.abspath(self.test_config)

        # split version
        if self.version:
            ver = self.version.split('.')
            self.version_major = int(ver[0])
            self.version_minor = int(ver[1])
            self.version_patch = int(ver[2])
            if(len(ver) == 4):
                self.version_build = int(ver[3])