def render(self, context):
        """inspect the code and look for files that can be turned into combos.
        Basically, the developer could type this:
        {% slimall %}
          <link href="/one.css"/>
          <link href="/two.css"/>
        {% endslimall %}
        And it should be reconsidered like this:
          <link href="{% slimfile "/one.css;/two.css" %}"/>
        which we already have routines for doing.
        """
        code = self.nodelist.render(context)
        if not settings.DJANGO_STATIC:
            # Append MEDIA_URL if set
            # quick and dirty
            if settings.DJANGO_STATIC_MEDIA_URL_ALWAYS:
                for match in STYLES_REGEX.finditer(code):
                    for filename in match.groups():
                        code = (code.replace(filename,
                                             settings.DJANGO_STATIC_MEDIA_URL + filename))

                for match in SCRIPTS_REGEX.finditer(code):
                    for filename in match.groups():
                        code = (code.replace(filename,
                                             settings.DJANGO_STATIC_MEDIA_URL + filename))

                return code

            return code

        new_js_filenames = []
        for match in SCRIPTS_REGEX.finditer(code):
            whole_tag = match.group()
            async_defer = ASYNC_DEFER_REGEX.search(whole_tag)
            for filename in match.groups():

                optimize_if_possible = self.optimize_if_possible
                if optimize_if_possible and \
                  (filename.endswith('.min.js') or filename.endswith('.minified.js')):
                    # Override! Because we simply don't want to run slimmer
                    # on files that have the file extension .min.js
                    optimize_if_possible = False

                new_js_filenames.append(filename)
                code = code.replace(whole_tag, '')

        # Now, we need to combine these files into one
        if new_js_filenames:
            new_js_filename = _static_file(new_js_filenames,
                               optimize_if_possible=optimize_if_possible,
                               symlink_if_possible=self.symlink_if_possible)
        else:
            new_js_filename = None

        new_image_filenames = []
        def image_replacer(match):
            tag = match.group()
            for filename in match.groups():
                new_filename = _static_file(filename,
                                            symlink_if_possible=self.symlink_if_possible)
                if new_filename != filename:
                    tag = tag.replace(filename, new_filename)
            return tag

        code = IMG_REGEX.sub(image_replacer, code)

        new_css_filenames = defaultdict(list)

        # It's less trivial with CSS because we can't combine those that are
        # of different media
        media_regex = re.compile('media=["\']([^"\']+)["\']')
        for match in STYLES_REGEX.finditer(code):
            whole_tag = match.group()
            try:
                media_type = media_regex.findall(whole_tag)[0]
            except IndexError:
                media_type = ''

            for filename in match.groups():
                new_css_filenames[media_type].append(filename)
                code = code.replace(whole_tag, '')

        # Now, we need to combine these files into one
        new_css_filenames_combined = {}
        if new_css_filenames:
            for media_type, filenames in new_css_filenames.items():
                r = _static_file(filenames,
                                 optimize_if_possible=self.optimize_if_possible,
                                 symlink_if_possible=self.symlink_if_possible)
                new_css_filenames_combined[media_type] = r


        if new_js_filename:
            # Now is the time to apply the name prefix if there is one
            if async_defer:
                new_tag = ('<script %s src="%s"></script>' %
                        (async_defer.group(0), new_js_filename))
            else:
                new_tag = '<script src="%s"></script>' % new_js_filename
            code = "%s%s" % (new_tag, code)

        for media_type, new_css_filename in new_css_filenames_combined.items():
            extra_params = ''
            if media_type:
                extra_params += ' media="%s"' % media_type
            new_tag = '<link rel="stylesheet"%s href="%s"/>' % \
              (extra_params, new_css_filename)
            code = "%s%s" % (new_tag, code)

        return code