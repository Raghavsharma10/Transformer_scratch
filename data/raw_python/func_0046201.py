def save_docs(self, files=None, output_dir=None):
        """
        Save documentation files for codebase into `output_dir`.  If output
        dir is None, it'll refrain from building the index page and build
        the file(s) in the current directory.

        If `files` is None, it'll build all files in the codebase.
        """
        if output_dir:
            try:
                os.mkdir(output_dir)
            except OSError:
                pass

            try:
                import pkg_resources
                save_file(os.path.join(output_dir, 'jsdoc.css'),
                    pkg_resources.resource_string(__name__, 'static/jsdoc.css'))
            except (ImportError, IOError):
                try:
                    import shutil
                    base_dir = os.path.dirname(os.path.realpath(__file__))
                    css_file = os.path.join(base_dir, 'jsdoc.css')
                    shutil.copy(css_file, output_dir)
                except IOError:
                    print('jsdoc.css not found.  HTML will not be styled.')

            save_file('%s/index.html' % output_dir, 
                    build_html_page('Module index', self.to_html()))
        else:
            output_dir = '.'

        if files is None:
            files = list(self.keys())

        for filename in files:
            try:
                doc = self[filename]
                save_file('%s/%s.html' % (output_dir, trim_js_ext(doc.name)), 
                        build_html_page(doc.name, doc.to_html(self)))
            except KeyError:
                warn('File %s does not exist', filename)