def create_gp(self):
        """Create a GnuPlot file for this BAM file."""

        categories_order = [
            ("{U}", "#ee82ee", 'Unmapped correctly'),
            ("{u}", "#ff0000", 'Unmapped incorrectly'),
            ("{T}", "#00ff00", 'Thresholded correctly'),
            ("{t}", "#008800", 'Thresholded incorrectly'),
            ("{P}", "#ffff00", 'Multimapped'),
            ("{w}+{x}", "#7f7f7f", 'Mapped, should be unmapped'),
            ("{m}", "#000000", 'Mapped to wrong position'),
            ("{M}", "#0000ff", 'Mapped correctly'),
        ]

        plot_lines = [
            '"{roc_fn}" using (( ({x}) )):({y}) lt rgb "{color}" with filledcurve x1 title "{title}", \\'.format(
                roc_fn=self._roc_fn,
                x=rnftools.lavender._format_xxx(self.default_x_axis),
                y=rnftools.lavender._format_xxx(
                    '({sum})*100/{{all}}'.format(sum="+".join([c[0] for c in categories_order[i:]]))
                ),
                color=categories_order[i][1],
                title=categories_order[i][2],
            ) for i in range(len(categories_order))
        ]

        plot = os.linesep.join((["plot \\"] + plot_lines + [""]))

        with open(self._gp_fn, "w+") as gp:
            gp_content = """
					set title "{{/:Bold=16 {title}}}"
	
					set x2lab "{x_lab}"
					set log x
					set log x2
	
					set format x "10^{{%L}}"
					set format x2 "10^{{%L}}"
					set xran  [{xran}]
					set x2ran [{xran}]
					set x2tics
					unset xtics
	
					set ylab "Part of all reads (%)"
	
					set format y "%g %%"
					set yran [{yran}]
					set y2ran [{yran}]
	
					set pointsize 1.5
	
					set grid ytics lc rgb "#777777" lw 1 lt 0 front
					set grid x2tics lc rgb "#777777" lw 1 lt 0 front
	
					set datafile separator "\\t"
					set palette negative
	
					set termin svg size {svg_size} enhanced
					set out "{svg_fn}"
					set key spacing 0.8 opaque width -5
				""".format(
                svg_fn=self._svg_fn,
                xran="{:.10f}:{:.10f}".format(self.report.default_x_run[0], self.report.default_x_run[1]),
                yran="{:.10f}:{:.10f}".format(self.report.default_y_run[0], self.report.default_y_run[1]),
                svg_size="{},{}".format(self.report.default_svg_size_px[0], self.report.default_svg_size_px[1]),
                title=os.path.basename(self._bam_fn)[:-4],
                x_lab=self.default_x_label,
            )
            gp_content = textwrap.dedent(gp_content) + "\n" + plot
            # gp_lines=gp_content.split("\n")
            # gp_lines=[x.strip() for x in gp_lines]
            # gp_content=gp_lines.join("\n")
            gp.write(gp_content)