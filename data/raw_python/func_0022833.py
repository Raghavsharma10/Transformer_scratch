def readLine(self):
        """ The method that reads a line and processes it.
        """

        # Read line
        line = self._f.readline().decode('ascii', 'ignore')
        if not line:
            raise EOFError()
        line = line.strip()

        if line.startswith('v '):
            # self._vertices.append( *self.readTuple(line) )
            self._v.append(self.readTuple(line))
        elif line.startswith('vt '):
            self._vt.append(self.readTuple(line, 3))
        elif line.startswith('vn '):
            self._vn.append(self.readTuple(line))
        elif line.startswith('f '):
            self._faces.append(self.readFace(line))
        elif line.startswith('#'):
            pass  # Comment
        elif line.startswith('mtllib '):
            logger.warning('Notice reading .OBJ: material properties are '
                           'ignored.')
        elif any(line.startswith(x) for x in ('g ', 's ', 'o ', 'usemtl ')):
            pass  # Ignore groups and smoothing groups, obj names, material
        elif not line.strip():
            pass
        else:
            logger.warning('Notice reading .OBJ: ignoring %s command.'
                           % line.strip())