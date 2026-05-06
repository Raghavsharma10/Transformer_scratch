def __set_log_file_name(self):
        """Automatically set logFileName attribute"""
        # ensure directory exists
        dir, _ = os.path.split(self.__logFileBasename)
        if len(dir) and not os.path.exists(dir):
            os.makedirs(dir)
        # create logFileName
        self.__logFileName = self.__logFileBasename+"."+self.__logFileExtension
        number = 0
        while os.path.isfile(self.__logFileName):
            if os.stat(self.__logFileName).st_size/1e6 < self.__maxlogFileSize:
                break
            number += 1
            self.__logFileName = self.__logFileBasename+"_"+str(number)+"."+self.__logFileExtension
        # create log file stram
        self.__logFileStream = None