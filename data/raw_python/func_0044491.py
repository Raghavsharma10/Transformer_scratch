def parse(self, fd):
        """very simple parser - but why would we want it to be complex?"""

        def resolve_args(args):
            # FIXME break this out, it's in common with the templating stuff elsewhere
            root = self.sections[0]
            val_dict = dict(('<' + t + '>', u) for (t, u) in root.get_variables().items())
            resolved_args = []
            for arg in args:
                for subst, value in val_dict.items():
                    arg = arg.replace(subst, value)
                resolved_args.append(arg)
            return resolved_args

        def handle_section_defn(keyword, parts):
            if keyword == '@HostAttrs':
                if len(parts) != 1:
                    raise ParserException('usage: @HostAttrs <hostname>')
                if self.sections[0].has_pending_with():
                    raise ParserException('@with not supported with @HostAttrs')
                self.sections.append(HostAttrs(parts[0]))
                return True
            if keyword == 'Host':
                if len(parts) != 1:
                    raise ParserException('usage: Host <hostname>')
                self.sections.append(Host(parts[0], self.sections[0].pop_pending_with()))
                return True

        def handle_vardef(root, keyword, parts):
            if keyword == '@with':
                root.add_pending_with(parts)
                return True

        def handle_set_args(_, parts):
            if len(parts) == 0:
                raise ParserException('usage: @args arg-name ...')
            if not self.is_include():
                return
            if self._args is None or len(self._args) != len(parts):
                raise ParserException('required arguments not passed to include {url} ({parts})'.format(
                    url=self._url,
                    parts=', '.join(parts))
                )
            root = self.sections[0]
            for key, value in zip(parts, self._args):
                root.set_value(key, value)

        def handle_set_value(_, parts):
            if len(parts) != 2:
                raise ParserException('usage: @set <key> <value>')
            root = self.sections[0]
            root.set_value(*resolve_args(parts))

        def handle_add_type(section, parts):
            if len(parts) != 1:
                raise ParserException('usage: @is <HostAttrName>')
            section.add_type(parts[0])

        def handle_via(section, parts):
            if len(parts) != 1:
                raise ParserException('usage: @via <Hostname>')
            section.add_line(
                'ProxyCommand',
                ('ssh {args} nc %h %p 2> /dev/null'.format(args=pipes.quote(resolve_args(parts)[0])), )
            )

        def handle_identity(section, parts):
            if len(parts) != 1:
                raise ParserException('usage: @identity <name>')
            section.add_identity(resolve_args(parts)[0])

        def handle_include(_, parts):
            if len(parts) == 0:
                raise ParserException('usage: @include <https://...|/path/to/file.sedge> [arg ...]')
            url = parts[0]
            parsed_url = urllib.parse.urlparse(url)
            if parsed_url.scheme == 'https':
                req = requests.get(url, verify=self._verify_ssl)
                text = req.text
            elif parsed_url.scheme == 'file':
                with open(parsed_url.path) as fd:
                    text = fd.read()
            elif parsed_url.scheme == '':
                path = os.path.expanduser(url)
                with open(path) as fd:
                    text = fd.read()
            else:
                raise SecurityException('error: @includes may only use paths or https:// or file:// URLs')

            subconfig = SedgeEngine(
                self._key_library,
                StringIO(text),
                self._verify_ssl,
                url=url,
                args=resolve_args(parts[1:]),
                parent_keydefs=self.keydefs,
                via_include=True)
            self.includes.append((url, subconfig))

        def handle_keydef(_, parts):
            if len(parts) < 2:
                raise ParserException('usage: @key <name> [fingerprint]...')
            name = parts[0]
            fingerprints = parts[1:]
            self.keydefs[name] = fingerprints

        def handle_keyword(section, keyword, parts):
            handlers = {
                '@set': handle_set_value,
                '@args': handle_set_args,
                '@is': handle_add_type,
                '@via': handle_via,
                '@include': handle_include,
                '@key': handle_keydef,
                '@identity': handle_identity
            }
            if keyword in handlers:
                handlers[keyword](section, parts)
                return True

        for line in (t.strip() for t in fd):
            if line.startswith('#') or line == '':
                continue
            keyword, parts = SedgeEngine.parse_config_line(line)
            if handle_section_defn(keyword, parts):
                continue
            if handle_vardef(self.sections[0], keyword, parts):
                continue
            current_section = self.sections[-1]
            if handle_keyword(current_section, keyword, parts):
                continue
            if keyword.startswith('@'):
                raise ParserException("unknown expansion keyword {}".format(keyword))
            # use other rather than parts to avoid messing up user
            # whitespace; we don't handle quotes in here as we don't
            # need to
            current_section.add_line(keyword, parts)