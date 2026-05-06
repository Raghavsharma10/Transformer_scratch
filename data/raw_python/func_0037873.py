def create_gp(self):
        """ Create GnuPlot file. """

        nb_bams = len(self.bams)

        gp_parts = [
            textwrap.dedent(
                """\
				set log x
				set log x2


				#set format x "10^{{%L}}"
				set format x2 "10^{{%L}}"
				set x2tics
				unset xtics
				"""
            ),
            os.linesep.join([self._gp_style_func(i, nb_bams) for i in range(nb_bams)]),
            textwrap.dedent(
                """\
					set format y "%g %%"
					set ytics

					set pointsize 1.5

					set grid ytics lc rgb "#777777" lw 1 lt 0 front
					set grid x2tics lc rgb "#777777" lw 1 lt 0 front

					set datafile separator "\\t"
					set palette negative
					"""
            ),
            os.linesep.join(self.gp_plots)
        ]

        gp_src = os.linesep.join(gp_parts)
        # .format(
        # 	x_lab=self.default_x_label,
        # )

        with open(self._gp_fn, "w+") as f:
            f.write(gp_src)