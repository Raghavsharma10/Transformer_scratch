def gen_code_api(self):
        """TODO: Docstring for gen_code_api."""

        # edit config file

        conf_editor = Editor(self.conf_fpath)

        # insert code path for searching
        conf_editor.editline_with_regex(r'^# import os', 'import os')
        conf_editor.editline_with_regex(r'^# import sys', 'import sys')
        conf_editor.editline_with_regex(
            r'^# sys\.path\.insert',
            'sys.path.insert(0, "{}")'.format(self.code_fdpath))
        conf_editor.editline_with_regex(
            r"""html_theme = 'alabaster'""",
            'html_theme = \'default\''.format(self.code_fdpath))

        conf_editor.finish_writing()

        # sphinx-apidoc to generate rst from source code

        # force regenerate
        subprocess.call(self._sphinx_apidoc_cmd)

        pass