def make_context(self, context, silent_build=False, extra_context=None):
        """
        Context manager for creating the context of the image

        Arguments:

        context - ``harpoon.option_spec.image_objs.Context``
            Knows all the context related options

        silent_build - boolean
            If True, then suppress printing out information

        extra_context - List of (content, string)
            content is either a string repsenting the content to put in a file
            or a dictionary representing what path to get from what docker image

            The second string represents where in the context this extra file should go
        """
        with a_temp_file() as tmpfile:
            t = tarfile.open(mode='w', fileobj=tmpfile)
            for thing, arcname in self.find_files_for_tar(context, silent_build):
                log.debug("Context: {0}".format(arcname))
                t.add(thing, arcname=arcname)

            if extra_context:
                extra = list(extra_context)
                for content, arcname in extra:
                    if arcname == "":
                        continue

                    with self.the_context(content, silent_build=silent_build) as fle:
                        log.debug("Context: {0}".format(arcname))
                        t.add(fle.name, arcname=arcname)

            yield ContextWrapper(t, tmpfile)