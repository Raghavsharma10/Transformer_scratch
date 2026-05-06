def generate(self, callback=None):
        """
        Computes and stores piece data. Returns ``True`` on success, ``False``
        otherwise.

        :param callback: progress/cancellation callable with method
            signature ``(filename, pieces_completed, pieces_total)``.
            Useful for reporting progress if dottorrent is used in a
            GUI/threaded context, and if torrent generation needs to be cancelled.
            The callable's return value should evaluate to ``True`` to trigger
            cancellation.
        """
        files = []
        single_file = os.path.isfile(self.path)
        if single_file:
            files.append((self.path, os.path.getsize(self.path), {}))
        elif os.path.exists(self.path):
            for x in os.walk(self.path):
                for fn in x[2]:
                    if any(fnmatch.fnmatch(fn, ext) for ext in self.exclude):
                        continue
                    fpath = os.path.normpath(os.path.join(x[0], fn))
                    fsize = os.path.getsize(fpath)
                    if fsize and not is_hidden_file(fpath):
                        files.append((fpath, fsize, {}))
        else:
            raise exceptions.InvalidInputException
        total_size = sum([x[1] for x in files])
        if not (len(files) and total_size):
            raise exceptions.EmptyInputException
        # set piece size if not already set
        if self.piece_size is None:
            self.piece_size = self.get_info()[2]
        if files:
            self._pieces = bytearray()
            i = 0
            num_pieces = math.ceil(total_size / self.piece_size)
            pc = 0
            buf = bytearray()
            while i < len(files):
                fe = files[i]
                f = open(fe[0], 'rb')
                if self.include_md5:
                    md5_hasher = md5()
                else:
                    md5_hasher = None
                for chunk in iter(lambda: f.read(self.piece_size), b''):
                    buf += chunk
                    if len(buf) >= self.piece_size \
                            or i == len(files)-1:
                        piece = buf[:self.piece_size]
                        self._pieces += sha1(piece).digest()
                        del buf[:self.piece_size]
                        pc += 1
                        if callback:
                            cancel = callback(fe[0], pc, num_pieces)
                            if cancel:
                                f.close()
                                return False
                    if self.include_md5:
                        md5_hasher.update(chunk)
                if self.include_md5:
                    fe[2]['md5sum'] = md5_hasher.hexdigest()
                f.close()
                i += 1
            # Add pieces from any remaining data
            while len(buf):
                piece = buf[:self.piece_size]
                self._pieces += sha1(piece).digest()
                del buf[:self.piece_size]
                pc += 1
                if callback:
                    cancel = callback(fe[0], pc, num_pieces)
                    if cancel:
                        return False

        # Create the torrent data structure
        data = OrderedDict()
        if len(self.trackers) > 0:
            data['announce'] = self.trackers[0].encode()
            if len(self.trackers) > 1:
                data['announce-list'] = [[x.encode()] for x in self.trackers]
        if self.comment:
            data['comment'] = self.comment.encode()
        if self.created_by:
            data['created by'] = self.created_by.encode()
        else:
            data['created by'] = DEFAULT_CREATOR.encode()
        if self.creation_date:
            data['creation date'] = int(self.creation_date.timestamp())
        if self.web_seeds:
            data['url-list'] = [x.encode() for x in self.web_seeds]
        data['info'] = OrderedDict()
        if single_file:
            data['info']['length'] = files[0][1]
            if self.include_md5:
                data['info']['md5sum'] = files[0][2]['md5sum']
            data['info']['name'] = files[0][0].split(os.sep)[-1].encode()
        else:
            data['info']['files'] = []
            path_sp = self.path.split(os.sep)
            for x in files:
                fx = OrderedDict()
                fx['length'] = x[1]
                if self.include_md5:
                    fx['md5sum'] = x[2]['md5sum']
                fx['path'] = [y.encode()
                              for y in x[0].split(os.sep)[len(path_sp):]]
                data['info']['files'].append(fx)
            data['info']['name'] = path_sp[-1].encode()
        data['info']['pieces'] = bytes(self._pieces)
        data['info']['piece length'] = self.piece_size
        data['info']['private'] = int(self.private)
        if self.source:
            data['info']['source'] = self.source.encode()

        self._data = data
        return True