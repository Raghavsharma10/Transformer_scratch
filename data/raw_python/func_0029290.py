def parse(self, inputstring, document):
        """Parse the nblink file.

        Adds the linked file as a dependency, read the file, and
        pass the content to the nbshpinx.NotebookParser.
        """
        link = json.loads(inputstring)
        env = document.settings.env
        source_dir = os.path.dirname(env.doc2path(env.docname))

        abs_path = os.path.normpath(os.path.join(source_dir, link['path']))
        path = utils.relative_path(None, abs_path)
        path = nodes.reprunicode(path)

        document.settings.record_dependencies.add(path)
        env.note_dependency(path)

        target_root = env.config.nbsphinx_link_target_root
        target = utils.relative_path(target_root, abs_path)
        target = nodes.reprunicode(target).replace(os.path.sep, '/')
        env.metadata[env.docname]['nbsphinx-link-target'] = target

        # Copy parser from nbsphinx for our cutom format
        try:
            formats = env.config.nbsphinx_custom_formats
        except AttributeError:
            pass
        else:
            formats.setdefault(
                '.nblink',
                lambda s: nbformat.reads(s, as_version=_ipynbversion))

        try:
            include_file = io.FileInput(source_path=path, encoding='utf8')
        except UnicodeEncodeError as error:
            raise NotebookError(u'Problems with linked notebook "%s" path:\n'
                                'Cannot encode input file path "%s" '
                                '(wrong locale?).' %
                                (env.docname, SafeString(path)))
        except IOError as error:
            raise NotebookError(u'Problems with linked notebook "%s" path:\n%s.' %
                                (env.docname, ErrorString(error)))

        try:
            rawtext = include_file.read()
        except UnicodeError as error:
            raise NotebookError(u'Problem with linked notebook "%s":\n%s' %
                                (env.docname, ErrorString(error)))
        return super(LinkedNotebookParser, self).parse(rawtext, document)