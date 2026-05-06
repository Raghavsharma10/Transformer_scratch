def magic(self, line):
        """
        Read and process magics
          @param line (str): the full line containing a magic
          @return (list): a tuple (output-message,css-class), where
            the output message can be a single string or a list (containing
            a Python format string and its arguments)
        """
        # The %lsmagic has no parameters
        if line.startswith('%lsmagic'):
            return magic_help, 'magic-help'

        # Split line into command & parameters
        try:
            cmd, param = line.split(None, 1)
        except ValueError:
            raise KrnlException("invalid magic: {}", line)
        cmd = cmd[1:].lower()

        # Process each magic
        if cmd == 'endpoint':

            self.srv = SPARQLWrapper.SPARQLWrapper(param)
            return ['Endpoint set to: {}', param], 'magic'

        elif cmd == 'auth':

            auth_data = param.split(None, 2)
            if auth_data[0].lower() == 'none':
                self.cfg.aut = None
                return ['HTTP authentication: None'], 'magic'
            if auth_data and len(auth_data) != 3:
                raise KrnlException("invalid %auth magic")
            self.cfg.aut = auth_data
            return ['HTTP authentication: {}', auth_data], 'magic'

        elif cmd == 'qparam':

            v = param.split(None, 1)
            if len(v) == 0:
                raise KrnlException("missing %qparam name")
            elif len(v) == 1:
                self.cfg.par.pop(v[0],None)
                return ['Param deleted: {}', v[0]]
            else:
                self.cfg.par[v[0]] = v[1]
                return ['Param set: {} = {}'] + v, 'magic'

        elif cmd == 'prefix':

            v = param.split(None, 1)
            if len(v) == 0:
                raise KrnlException("missing %prefix value")
            elif len(v) == 1:
                self.cfg.pfx.pop(v[0], None)
                return ['Prefix deleted: {}', v[0]], 'magic'
            else:
                self.cfg.pfx[v[0]] = v[1]
                return ['Prefix set: {} = {}'] + v, 'magic'

        elif cmd == 'show':

            if param == 'all':
                self.cfg.lmt = None
            else:
                try:
                    self.cfg.lmt = int(param)
                except ValueError as e:
                    raise KrnlException("invalid result limit: {}", e)
            sz = self.cfg.lmt if self.cfg.lmt is not None else 'unlimited'
            return ['Result maximum size: {}', sz], 'magic'

        elif cmd == 'format':

            fmt_list = {'JSON': SPARQLWrapper.JSON, 
                        'N3': SPARQLWrapper.N3,
                        'XML': SPARQLWrapper.XML,
                        'DEFAULT': None,
                        'ANY': False}
            try:
                fmt = param.upper()
                self.cfg.fmt = fmt_list[fmt]
            except KeyError:
                raise KrnlException('unsupported format: {}\nSupported formats are: {!s}', param, list(fmt_list.keys()))
            return ['Return format: {}', fmt], 'magic'

        elif cmd == 'lang':

            self.cfg.lan = DEFAULT_TEXT_LANG if param == 'default' else [] if param=='all' else param.split()
            return ['Label preferred languages: {}', self.cfg.lan], 'magic'

        elif cmd in 'graph':

            self.cfg.grh = param if param else None
            return ['Default graph: {}', param if param else 'None'], 'magic'

        elif cmd == 'display':

            v = param.lower().split(None, 2)
            if len(v) == 0 or v[0] not in ('table', 'raw', 'graph', 'diagram'):
                raise KrnlException('invalid %display command: {}', param)

            msg_extra = ''
            if v[0] not in ('diagram', 'graph'):
                self.cfg.dis = v[0]
                self.cfg.typ = len(v) > 1 and v[1].startswith('withtype')
                if self.cfg.typ and self.cfg.dis == 'table':
                    msg_extra = '\nShow Types: on'
            elif len(v) == 1:   # graph format, defaults
                self.cfg.dis = ['svg']
            else:               # graph format, with options
                if v[1] not in ('png', 'svg'):
                    raise KrnlException('invalid graph format: {}', param)
                if len(v) > 2:
                    if not v[2].startswith('withlit'):
                        raise KrnlException('invalid graph option: {}', param)
                    msg_extra = '\nShow literals: on'
                self.cfg.dis = v[1:3]

            display = self.cfg.dis[0] if is_collection(self.cfg.dis) else self.cfg.dis
            return ['Display: {}{}', display, msg_extra], 'magic'

        elif cmd == 'outfile':

            if param == 'NONE':
                self.cfg.out = None
                return ['no output file'], 'magic'
            else:
                self.cfg.out = param
                return ['Output file: {}', os.path.abspath(param)], 'magic'

        elif cmd == 'log':

            if not param:
                raise KrnlException('missing log level')
            try:
                lev = param.upper()
                parent_logger = logging.getLogger(__name__.rsplit('.', 1)[0])
                parent_logger.setLevel(lev)
                return ("Logging set to {}", lev), 'magic'
            except ValueError:
                raise KrnlException('unknown log level: {}', param)

        elif cmd == 'header':

            if param.upper() == 'OFF':
                num = len(self.cfg.hdr)
                self.cfg.hdr = []
                return ['All headers deleted ({})', num], 'magic'
            else:
                if param in self.cfg.hdr:
                    return ['Header skipped (repeated)'], 'magic'
                self.cfg.hdr.append(param)
                return ['Header added: {}', param], 'magic'

        else:
            raise KrnlException("magic not found: {}", cmd)