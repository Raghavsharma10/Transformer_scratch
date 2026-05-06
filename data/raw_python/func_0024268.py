def _extract_translations(self, domains):
        """Extract the translations into `.pot` files"""
        for domain, options in domains.items():
            # Create the extractor
            extractor = babel_frontend.extract_messages()
            extractor.initialize_options()
            # The temporary location to write the `.pot` file
            extractor.output_file = options['pot']
            # Add the comments marked with 'tn:' to the translation file for translators to read. Strip the marker.
            extractor.add_comments = ['tn:']
            extractor.strip_comments = True
            # The directory where the sources for this domain are located
            extractor.input_paths = [options['source']]
            # Pass the metadata to the translator
            extractor.msgid_bugs_address = self.manager.args.contact
            extractor.copyright_holder = self.manager.args.copyright
            extractor.version = self.manager.args.version
            extractor.project = self.manager.args.project
            extractor.finalize_options()
            # Add keywords for lazy translation functions, based on their non-lazy variants
            extractor.keywords.update({
                'gettext_lazy': extractor.keywords['gettext'],
                'ngettext_lazy': extractor.keywords['ngettext'],
                '__': extractor.keywords['gettext'],  # double underscore for lazy
            })
            # Do the extraction
            _run_babel_command(extractor)