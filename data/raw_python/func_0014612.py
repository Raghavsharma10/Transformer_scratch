def register_upload_callback(self, *args, **kwargs):
        """Registers an Upload function (see :ref:`upload-plugin`)
        to handle a certain form.

        Refer to :func:`sijax.plugin.upload.register_upload_callback`
        for more details.

        This method passes some additional arguments to your handler
        functions - the ``flask.request.files`` object.

        Your upload handler function's signature should look like this::

            def func(obj_response, files, form_values)

        :return: string - javascript code that initializes the form
        """
        if 'args_extra' not in kwargs:
            kwargs['args_extra'] = [request.files]
        return sijax.plugin.upload.register_upload_callback(self._sijax, *args, **kwargs)