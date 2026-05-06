def query(self, query, num=0, silent=False):
        """
        Launch an SPARQL query, process & convert results and return them
        """
        if self.srv is None:
            raise KrnlException('no endpoint defined')

        # Add to the query all predefined SPARQL prefixes
        if self.cfg.pfx:
            prefix = '\n'.join(('PREFIX {} {}'.format(*v)
                                for v in self.cfg.pfx.items()))
            query = prefix + '\n' + query

        # Prepend to the query all predefined Header entries
        # The header should be before the prefix and other sparql commands
        if self.cfg.hdr:
            query = '\n'.join(self.cfg.hdr) + '\n' + query

        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug("\n%50s%s", query, '...' if len(query) > 50 else '')

        # Select requested format
        if self.cfg.fmt is not None:
            fmt_req = self.cfg.fmt
        elif re.search(r'\bselect\b', query, re.I):
            fmt_req = SPARQLWrapper.JSON
        elif re.search(r'\b(?:describe|construct)\b', query, re.I):
            fmt_req = SPARQLWrapper.N3
        else:
            fmt_req = False

        # Set the query
        self.srv.resetQuery()
        if self.cfg.aut:
            self.srv.setHTTPAuth(self.cfg.aut[0])
            self.srv.setCredentials(*self.cfg.aut[1:])
        else:
            self.srv.setCredentials(None, None)
        self.log.debug(u'request-format: %s  display: %s', fmt_req, self.cfg.dis)
        if fmt_req:
            self.srv.setReturnFormat(fmt_req)
        if self.cfg.grh:
            self.srv.addParameter("default-graph-uri", self.cfg.grh)
        for p in self.cfg.par.items():
            self.srv.addParameter(*p)
        self.srv.setQuery(query)

        if not silent or self.cfg.out:
            try:
                # Launch query
                start = datetime.datetime.utcnow()
                res = self.srv.query()
                now = datetime.datetime.utcnow()
                self.log.debug(u'response elapsed=%s', now-start)
                start = now

                # See what we got
                info = res.info()
                self.log.debug(u'response info: %s', info)
                fmt_got = info['content-type'].split(';')[0] if 'content-type' in info else None

                # Check we received a MIME type according to what we requested
                if fmt_req and fmt_got not in mime_type[fmt_req]:
                    raise KrnlException(u'Unexpected response format: {} (requested: {})', fmt_got, fmt_req)

                # Get the result
                data = b''.join((line for line in res))

            except KrnlException:
                raise
            except SPARQLWrapperException as e:
                raise KrnlException(u'SPARQL error: {}', touc(e))
            except Exception as e:
                raise KrnlException(u'Query processing error: {!s}', e)

            # Write the raw result to a file
            if self.cfg.out:
                try:
                    outname = self.cfg.out % num
                except TypeError:
                    outname = self.cfg.out
                with io.open(outname, 'wb') as f:
                    f.write(data)

            # Render the result into the desired display format
            try:
                # Data format we will render
                fmt = (fmt_req if fmt_req else
                       SPARQLWrapper.JSON if fmt_got in mime_type[SPARQLWrapper.JSON] else
                       SPARQLWrapper.N3 if fmt_got in mime_type[SPARQLWrapper.N3] else
                       SPARQLWrapper.XML if fmt_got in mime_type[SPARQLWrapper.XML] else
                       'text/plain' if self.cfg.dis == 'raw' else
                       fmt_got if fmt_got in ('text/plain', 'text/html') else
                       'text/plain')
                #self.log.debug(u'format: req=%s got=%s rend=%s',fmt_req,fmt_got,fmt)

                # Can't process? Just write the data as is
                if fmt in ('text/plain', 'text/html'):
                    out = data.decode('utf-8') if isinstance(data, bytes) else data
                    r = {'data': {fmt: out}, 'metadata': {}}
                else:
                    f = render_json if fmt == SPARQLWrapper.JSON else render_xml if fmt == SPARQLWrapper.XML else render_graph
                    r = f(data, self.cfg, format=fmt_got)
                    now = datetime.datetime.utcnow()
                    self.log.debug(u'response formatted=%s', now-start)
                if not silent:
                    return r

            except Exception as e:
                raise KrnlException(u'Response processing error: {}', touc(e))