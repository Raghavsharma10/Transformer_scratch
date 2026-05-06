def _parse(self, command):
        """ Parse a single command.
        """
        cmd, id_, args = command[0], command[1], command[2:]

        if cmd == 'CURRENT':
            # This context is made current
            self.env.clear()
            self._gl_initialize()
            self.env['fbo'] = args[0]
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, args[0])
        elif cmd == 'FUNC':
            # GL function call
            args = [as_enum(a) for a in args]
            try:
                getattr(gl, id_)(*args)
            except AttributeError:
                logger.warning('Invalid gl command: %r' % id_)
        elif cmd == 'CREATE':
            # Creating an object
            if args[0] is not None:
                klass = self._classmap[args[0]]
                self._objects[id_] = klass(self, id_)
            else:
                self._invalid_objects.add(id_)
        elif cmd == 'DELETE':
            # Deleting an object
            ob = self._objects.get(id_, None)
            if ob is not None:
                self._objects[id_] = JUST_DELETED
                ob.delete()
        else:
            # Doing somthing to an object
            ob = self._objects.get(id_, None)
            if ob == JUST_DELETED:
                return
            if ob is None:
                if id_ not in self._invalid_objects:
                    raise RuntimeError('Cannot %s object %i because it '
                                       'does not exist' % (cmd, id_))
                return
            # Triage over command. Order of commands is set so most
            # common ones occur first.
            if cmd == 'DRAW':  # Program
                ob.draw(*args)
            elif cmd == 'TEXTURE':  # Program
                ob.set_texture(*args)
            elif cmd == 'UNIFORM':  # Program
                ob.set_uniform(*args)
            elif cmd == 'ATTRIBUTE':  # Program
                ob.set_attribute(*args)
            elif cmd == 'DATA':  # VertexBuffer, IndexBuffer, Texture
                ob.set_data(*args)
            elif cmd == 'SIZE':  # VertexBuffer, IndexBuffer,
                ob.set_size(*args)  # Texture[1D, 2D, 3D], RenderBuffer
            elif cmd == 'ATTACH':  # FrameBuffer
                ob.attach(*args)
            elif cmd == 'FRAMEBUFFER':  # FrameBuffer
                ob.set_framebuffer(*args)
            elif cmd == 'SHADERS':  # Program
                ob.set_shaders(*args)
            elif cmd == 'WRAPPING':  # Texture1D, Texture2D, Texture3D
                ob.set_wrapping(*args)
            elif cmd == 'INTERPOLATION':  # Texture1D, Texture2D, Texture3D
                ob.set_interpolation(*args)
            else:
                logger.warning('Invalid GLIR command %r' % cmd)