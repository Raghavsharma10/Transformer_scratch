async def parseform(self, limit = 67108864, tostr = True, safename = True):
        '''
        Parse form-data with multipart/form-data or application/x-www-form-urlencoded
        In Python3, the keys of form and files are unicode, but values are bytes
        If the key ends with '[]', it is considered to be a list:
        a=1&b=2&b=3          =>    {'a':1,'b':3}
        a[]=1&b[]=2&b[]=3    =>    {'a':[1],'b':[2,3]}
        :param limit: limit total input size, default to 64MB. None = no limit. Note that all the form
        data is stored in memory (including upload files), so it is dangerous to accept a very large input.
        :param tostr: convert values to str in Python3. Only apply to form, files data are always bytes
        :param safename: if True, extra security checks are performed on filenames to reduce known security risks.
        '''
        if tostr:
            def _str(s):
                try:
                    if not isinstance(s, str):
                        return s.decode(self.encoding)
                    else:
                        return s
                except Exception:
                    raise HttpInputException('Invalid encoding in post data: ' + repr(s))
        else:
            def _str(s):
                return s
        try:
            form = {}
            files = {}
            # If there is not a content-type header, maybe there is not a content.
            if b'content-type' in self.headerdict and self.inputstream is not None:
                contenttype = self.headerdict[b'content-type']
                m = Message()
                # Email library expects string, which is unicode in Python 3
                try:
                    m.add_header('Content-Type', str(contenttype.decode('ascii')))
                except UnicodeDecodeError:
                    raise HttpInputException('Content-Type has non-ascii characters')
                if m.get_content_type() == 'multipart/form-data':
                    fp = BytesFeedParser()
                    fp.feed(b'Content-Type: ' + contenttype + b'\r\n\r\n')
                    total_length = 0
                    while True:
                        try:
                            await self.inputstream.prepareRead(self.container)
                            data = self.inputstream.readonce()
                            total_length += len(data)
                            if limit is not None and total_length > limit:
                                raise HttpInputException('Data is too large')
                            fp.feed(data)
                        except EOFError:
                            break
                    msg = fp.close()
                    if not msg.is_multipart() or msg.defects:
                        # Reject the data
                        raise HttpInputException('Not valid multipart/form-data format')
                    for part in msg.get_payload():
                        if part.is_multipart() or part.defects:
                            raise HttpInputException('Not valid multipart/form-data format')
                        disposition = part.get_params(header='content-disposition')
                        if not disposition:
                            raise HttpInputException('Not valid multipart/form-data format')
                        disposition = dict(disposition)
                        if 'form-data' not in disposition or 'name' not in disposition:
                            raise HttpInputException('Not valid multipart/form-data format')
                        if 'filename' in disposition:
                            name = disposition['name']
                            filename = disposition['filename']
                            if safename:
                                filename = _safename(filename)
                            if name.endswith('[]'):
                                files.setdefault(name[:-2], []).append({'filename': filename, 'content': part.get_payload(decode=True)})
                            else:
                                files[name] = {'filename': filename, 'content': part.get_payload(decode=True)}
                        else:
                            name = disposition['name']
                            if name.endswith('[]'):
                                form.setdefault(name[:-2], []).append(_str(part.get_payload(decode=True)))
                            else:
                                form[name] = _str(part.get_payload(decode=True))
                elif m.get_content_type() == 'application/x-www-form-urlencoded' or \
                        m.get_content_type() == 'application/x-url-encoded':
                    if limit is not None:
                        data = await self.inputstream.read(self.container, limit + 1)
                        if len(data) > limit:
                            raise HttpInputException('Data is too large')
                    else:
                        data = await self.inputstream.read(self.container)
                    result = parse_qs(data, True)
                    def convert(k,v):
                        try:
                            k = str(k.decode('ascii'))
                        except Exception:
                            raise HttpInputException('Form-data key must be ASCII')
                        if not k.endswith('[]'):
                            v = _str(v[-1])
                        else:
                            k = k[:-2]
                            v = [_str(i) for i in v]
                        return (k,v)
                    form = dict(convert(k,v) for k,v in result.items())
                else:
                    # Other formats, treat like no data
                    pass
            self.form = form
            self.files = files                
        except Exception as exc:
            raise HttpInputException('Failed to parse form-data: ' + str(exc))