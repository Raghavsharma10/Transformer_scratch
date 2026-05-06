def _render_edf(self, orig_tex):
        """Render an EDF to a texture"""
        # Set up the necessary textures
        sdf_size = orig_tex.shape[:2]

        comp_texs = []
        for _ in range(2):
            tex = Texture2D(sdf_size + (4,), format='rgba',
                            interpolation='nearest', wrapping='clamp_to_edge')
            comp_texs.append(tex)
        self.fbo_to[0].color_buffer = comp_texs[0]
        self.fbo_to[1].color_buffer = comp_texs[1]
        for program in self.programs[1:]:  # program_seed does not need this
            program['u_texh'], program['u_texw'] = sdf_size

        # Do the rendering
        last_rend = 0
        with self.fbo_to[last_rend]:
            set_viewport(0, 0, sdf_size[1], sdf_size[0])
            self.program_seed['u_texture'] = orig_tex
            self.program_seed.draw('triangle_strip')
        stepsize = (np.array(sdf_size) // 2).max()
        while stepsize > 0:
            self.program_flood['u_step'] = stepsize
            self.program_flood['u_texture'] = comp_texs[last_rend]
            last_rend = 1 if last_rend == 0 else 0
            with self.fbo_to[last_rend]:
                set_viewport(0, 0, sdf_size[1], sdf_size[0])
                self.program_flood.draw('triangle_strip')
            stepsize //= 2
        return comp_texs[last_rend]