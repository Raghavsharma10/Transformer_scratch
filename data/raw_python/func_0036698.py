def add_logger(self, cb, level='NORMAL', filters='ALL'):
        '''Add a callback to receive log events from this component.

        @param cb The callback function to receive log events. It must have the
            signature cb(name, time, source, level, message), where name is the
            name of the component the log record came from, time is a
            floating-point time stamp, source is the name of the logger that
            provided the log record, level is the log level of the record and
            message is a text string.
        @param level The maximum level of log records to receive.
        @param filters Filter the objects from which to receive log messages.
        @return An ID for this logger. Use this ID in future operations such as
                removing this logger.
        @raises AddLoggerError

        '''
        with self._mutex:
            obs = sdo.RTCLogger(self, cb)
            uuid_val = uuid.uuid4()
            intf_type = obs._this()._NP_RepositoryId
            props = {'logger.log_level': level,
                    'logger.filter': filters}
            props = utils.dict_to_nvlist(props)
            sprof = SDOPackage.ServiceProfile(id=uuid_val.get_bytes(),
                    interface_type=intf_type, service=obs._this(),
                    properties=props)
            conf = self.object.get_configuration()
            res = conf.add_service_profile(sprof)
            if res:
                self._loggers[uuid_val] = obs
                return uuid_val
            raise exceptions.AddLoggerError(self.name)