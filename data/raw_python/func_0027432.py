def input(self, _in, out, **kwargs):
        """Process individual translation file."""
        language_code = _re_language_code.search(_in.read()).group(
            'language_code'
        )
        _in.seek(0)  # move at the begining after matching the language
        catalog = read_po(_in)
        out.write('gettextCatalog.setStrings("{0}", '.format(language_code))
        out.write(json.dumps({
            key: value.string for key, value in catalog._messages.items()
            if key and value.string
        }))
        out.write(');')