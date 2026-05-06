def get_at_url(self, url):
        """
        Return a object representing the content at url.

        Returns None if no object could be matched with the id.

        Works for Album, Comment, Gallery_album, Gallery_image, Image and User.

        NOTE: Imgur's documentation does not cover what urls are available.
        Some urls, such as imgur.com/<ID> can be for several different types of
        object. Using a wrong, but similair call, such as get_subreddit_image
        on a meme image will not cause an error. But instead return a subset of
        information, with either the remaining pieces missing or the value set
        to None. This makes it hard to create a method such as this that
        attempts to deduce the object from the url. Due to these factors, this
        method should be considered experimental and used carefully.

        :param url: The url where the content is located at
        """
        class NullDevice():
            def write(self, string):
                pass

        def get_gallery_item(id):
            """
            Special helper method to get gallery items.

            The problem is that it's impossible to distinguish albums and
            images from each other based on the url. And there isn't a common
            url endpoints that return either a Gallery_album or a Gallery_image
            depending on what the id represents. So the only option is to
            assume it's a Gallery_image and if we get an exception then try
            Gallery_album.  Gallery_image is attempted first because there is
            the most of them.
            """
            try:
                # HACK: Problem is that send_request prints the error message
                # from Imgur when it encounters an error. This is nice because
                # this error message is more descriptive than just the status
                # code that Requests give. But since we first assume the id
                # belong to an image, it means we will get an error whenever
                # the id belongs to an album. The following code temporarily
                # disables stdout to avoid give a cryptic and incorrect error.

                # Code for disabling stdout is from
                # http://coreygoldberg.blogspot.dk/2009/05/
                # python-redirect-or-turn-off-stdout-and.html
                original_stdout = sys.stdout  # keep a reference to STDOUT
                sys.stdout = NullDevice()  # redirect the real STDOUT
                return self.get_gallery_image(id)
            # TODO: Add better error codes so I don't have to do a catch-all
            except Exception:
                return self.get_gallery_album(id)
            finally:
                sys.stdout = original_stdout  # turn STDOUT back on

        if not self.is_imgur_url(url):
            return None

        objects = {'album': {'regex': "a/(?P<id>[\w.]*?)$",
                             'method': self.get_album},
                   'comment': {'regex': "gallery/\w*/comment/(?P<id>[\w.]*?)$",
                               'method': self.get_comment},
                   'gallery': {'regex': "(gallery|r/\w*?)/(?P<id>[\w.]*?)$",
                               'method': get_gallery_item},
                   # Valid image extensions: http://imgur.com/faq#types
                   # All are between 3 and 4 chars long.
                   'image': {'regex': "(?P<id>[\w.]*?)(\\.\w{3,4})?$",
                             'method': self.get_image},
                   'user': {'regex': "user/(?P<id>[\w.]*?)$",
                            'method': self.get_user}
                   }
        parsed_url = urlparse(url)
        for obj_type, values in objects.items():
            regex_result = re.match('/' + values['regex'], parsed_url.path)
            if regex_result is not None:
                obj_id = regex_result.group('id')
                initial_object = values['method'](obj_id)
                if obj_type == 'image':
                    try:
                        # A better version might be to ping the url where the
                        # gallery_image should be with a requests.head call. If
                        # we get a 200 returned, then that means it exists and
                        # this becomes less hacky.
                        original_stdout = sys.stdout
                        sys.stdout = NullDevice()
                        if getattr(initial_object, 'section', None):
                            sub = initial_object.section
                            return self.get_subreddit_image(sub, obj_id)
                        return self.get_gallery_image(obj_id)
                    except Exception:
                        pass
                    finally:
                        sys.stdout = original_stdout
                return initial_object