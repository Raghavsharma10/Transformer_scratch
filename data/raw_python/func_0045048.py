def add_controls(self, env, target_name='control',
                     file_name='control.json',
                     encoder_cls=SConsEncoder):
        """
        Adds a target to build a control file at each of the current leaves.

        :param env: SCons Environment object
        :param target_name: Name for target in nest
        :param file_name: Name for output file.
        """
        if not HAS_SCONS:
            raise ImportError('SCons not available')

        @self.add_target(name=target_name)
        def control(outdir, c):
            return env.Command(os.path.join(outdir, file_name),
                               [],
                               action=_create_control_file,
                               control_dict=c,
                               encoder_cls=encoder_cls)