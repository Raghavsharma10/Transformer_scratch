def get_traceback_data(self):
        """Return a dictionary containing traceback information."""
        default_template_engine = None

        if default_template_engine is None:
            template_loaders = []

        frames = self.get_traceback_frames()
        for i, frame in enumerate(frames):
            if 'vars' in frame:
                frame_vars = []
                for k, v in frame['vars']:
                    v = pformat(v)
                    # The escape filter assume unicode, make sure that works
                    if isinstance(v, six.binary_type):
                        v = v.decode('utf-8', 'replace')  # don't choke on non-utf-8 input
                    # Trim large blobs of data
                    if v and len(v) > 4096:
                        v = '%s... <trimmed %d bytes string>' % (v[0:4096], len(v))
                    frame_vars.append((k, v))
                frame['vars'] = frame_vars
            frames[i] = frame

        unicode_hint = ''
        if self.exc_type and issubclass(self.exc_type, UnicodeError):
            start = getattr(self.exc_value, 'start', None)
            end = getattr(self.exc_value, 'end', None)
            if start is not None and end is not None:
                unicode_str = self.exc_value.args[1]

        c = {
            'is_email': False,
            'frames': frames,
            'sys_executable': sys.executable,
            'sys_version_info': '%d.%d.%d' % sys.version_info[0:3],
            'sys_path': sys.path,
        }

        # Check whether exception info is available
        if self.exc_type:
            c['exception_type'] = self.exc_type.__name__
        if self.exc_value:
            c['exception_value'] = self.exc_value
        if frames:
            c['lastframe'] = frames[-1]
        return c