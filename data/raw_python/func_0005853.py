def get_log_format_default(self):
        """Returns default log message format.

        .. note:: Some params may be missing.

        """
        vars = self.logging.vars

        format_default = (
            '[pid: %s|app: %s|req: %s/%s] %s (%s) {%s vars in %s bytes} [%s] %s %s => '
            'generated %s bytes in %s %s%s(%s %s) %s headers in %s bytes (%s switches on core %s)' % (

                vars.WORKER_PID,
                '-',  # app id
                '-',  # app req count
                '-',  # worker req count
                vars.REQ_REMOTE_ADDR,
                vars.REQ_REMOTE_USER,
                vars.REQ_COUNT_VARS_CGI,
                vars.SIZE_PACKET_UWSGI,
                vars.REQ_START_CTIME,
                vars.REQ_METHOD,
                vars.REQ_URI,
                vars.RESP_SIZE_BODY,
                vars.RESP_TIME_MS,  # or RESP_TIME_US,
                '-',  # tsize
                '-',  # via sendfile/route/offload
                vars.REQ_SERVER_PROTOCOL,
                vars.RESP_STATUS,
                vars.RESP_COUNT_HEADERS,
                vars.RESP_SIZE_HEADERS,
                vars.ASYNC_SWITCHES,
                vars.CORE,
        ))

        return format_default