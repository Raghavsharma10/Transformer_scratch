def update_log_type(self, logType, name=None, level=None, stdoutFlag=None, fileFlag=None, color=None, highlight=None, attributes=None):
        """
        update a logtype.

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
        assert logType in self.__logTypeStdoutFlags.keys(), "logType '%s' is not defined" %logType
        # get None updates
        if name is None:       name       = self.__logTypeNames[logType]
        if level is None:      level      = self.__logTypeLevels[logType]
        if stdoutFlag is None: stdoutFlag = self.__logTypeStdoutFlags[logType]
        if fileFlag is None:   fileFlag   = self.__logTypeFileFlags[logType]
        if color is None:      color      = self.__logTypeColor[logType]
        if highlight is None:  highlight  = self.__logTypeHighlight[logType]
        if attributes is None: attributes = self.__logTypeAttributes[logType]
        # update log type
        self.__set_log_type(logType=logType, name=name, level=level,
                            stdoutFlag=stdoutFlag, fileFlag=fileFlag,
                            color=color, highlight=highlight, attributes=attributes)