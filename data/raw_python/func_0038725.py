def create_html(self):
        """Create HTML report."""

        html_table = ""
        columns = [panel.get_html_column() for panel in self.panels]
        trs = len(columns[0])
        html_table += os.linesep.join(
            [
                "<tr>{}</tr>".format("".join(["<td>{}</td>".format(columns[col][row]) for col in range(len(columns))]))
                for row in range(trs)
            ]
        )

        with open(self._html_fn, "w+") as f:
            css_src = textwrap.dedent(
                """\
					.main_table                       {border-collapse:collapse;margin-top:15px;}
					td                                {border: solid #aaaaff 1px;padding:4px;vertical-alignment:top;}
					colgroup, thead                   {border: solid black 2px;padding 2px;}
					.configuration                    {font-size:85%;}
					.configuration, .configuration *  {margin:0;padding:0;}
					.formats                          {text-align:center;margin:20px 0px;}
					img                               {min-width:640px}
			"""
            )

            html_src = """<!DOCTYPE html>
			<html>
			<head>
				<meta charset="UTF-8" />
				<title>{title}</title>
				<style type="text/css">
				{css}
				</style>
			</head>
			<body>
				<h1>{title}</h1>
				<strong>{description}</strong>

				<table class="main_table">
				{html_table}
				</table>

			</body>
			""".format(
                html_table=html_table,
                css=css_src,
                title=self.title,
                description=self.description,
            )

            tidy_html_src = bs4.BeautifulSoup(html_src).prettify()
            f.write(tidy_html_src)