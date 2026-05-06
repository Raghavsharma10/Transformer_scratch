def add_log_type(self, logType, name=None, level=0, stdoutFlag=None, fileFlag=None, color=None, highlight=None, attributes=None):
        """
        Add a new logtype.

        :Parameters:
           #. logType (string): The logtype.
           #. name (None, string): The logtype name. If None, name will be set to logtype.
           #. level (number): The level of logging.
           #. stdoutFlag (None, boolean): Force standard output logging flag.
              If None, flag will be set according to minimum and maximum levels.
           #. fileFlag (None, boolean): Force file logging flag.
              If None, flag will be set according to minimum and maximum levels.
           #. color (None, string): The logging text color. The defined colors are:\n
              black , red , green , orange , blue , magenta , cyan , grey , dark grey ,
              light red , light green , yellow , light blue , pink , light cyan
           #. highlight (None, string): The logging text highlight color. The defined highlights are:\n
              black , red , green , orange , blue , magenta , cyan , grey
           #. attributes (None, string): The logging text attribute. The defined attributes are:\n
              bold , underline , blink , invisible , strike through

        **N.B** *logging color, highlight and attributes are not allowed on all types of streams.*
        """
        # check logType
        assert logType not in self.__logTypeStdoutFlags.keys(), "logType '%s' already defined" %logType
        assert isinstance(logType, basestring), "logType must be a string"
        logType = str(logType)
        # set log type
        self.__set_log_type(logType=logType, name=name, level=level,
                            stdoutFlag=stdoutFlag, fileFlag=fileFlag,
                            color=color, highlight=highlight, attributes=attributes)