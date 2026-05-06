def get_html_column(self):
        """ Get a HTML column for this panel. """

        panel_id = "panel_{}".format(self.name)
        return ["<h2>{}</h2>".format(self.title) + '<a href="{}">Download data</a>'.format(self.tar_fn())] + [
            # list of links
            (" <br />" + os.linesep).join(
                [
                    """
						   <strong>{bam_name}:</strong>
						   <a onclick="document.getElementById('{panel_id}').src='{bam_svg}';document.getElementById('{panel_id}_').href='{bam_html}';return false;" href="#">display graph</a>,
						   <a href="{bam_html}">detailed report</a>
						   """.format(
                        bam_name=bam.get_name(),
                        bam_html=bam.html_fn(),
                        bam_svg=bam.svg_fn(),
                        panel_id=panel_id,
                    ) for bam in self.bams
                ]
            ) + '<br /> '.format(self.tar_fn()),

            # main graph
            """
				   <div class="formats">
				   <a href="{html}" id="{panel_id}_">
				   <img src="{svg}" id="{panel_id}" />
				   </a>
				   </div>
				   """.format(
                html=self.bams[0]._html_fn,
                svg=self.bams[0]._svg_fn,
                panel_id=panel_id,
            ),
        ] + [
            # overall graphs
            """
				   <div class="formats">
				   <img src="{svg}" />
				   <br />
				   <a href="{svg}">SVG version</a>
				   |
				   <a href="{gp}" type="text/plain">GP file</a>
				   </div>
				   
				   """.format(
                svg=svg,
                gp=self._gp_fn,
            ) for svg in self._svg_fns
        ]