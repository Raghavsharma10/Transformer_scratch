def send(self, auto_complete=True, callback=None):
        """Begin uploading file(s) and sending email(s).
        If `auto_complete` is set to ``False`` you will have to call the
        :func:`Transfer.complete` function at a later stage.

        :param auto_complete: Whether or not to mark transfer as complete
         and send emails to recipient(s)
        :param callback: Callback function which will receive total file size
         and bytes read as arguments
        :type auto_complete: ``bool``
        :type callback: ``func``
        """

        tot = len(self.files)
        url = self.transfer_info['transferurl']

        for index, fmfile in enumerate(self.files):

            msg = 'Uploading: "{filename}" ({cur}/{tot})'
            logger.debug(
                msg.format(
                    filename=fmfile['thefilename'],
                    cur=index + 1,
                    tot=tot)
                )

            with open(fmfile['filepath'], 'rb') as file_obj:
                fields = {
                    fmfile['thefilename']: (
                        'filename',
                        file_obj,
                        fmfile['content-type']
                        )
                    }

                def pg_callback(monitor):
                    if pm.COMMANDLINE:
                        bar.show(monitor.bytes_read)

                    elif callback is not None:
                        callback(fmfile['totalsize'], monitor.bytes_read)

                m_encoder = encoder.MultipartEncoder(fields=fields)
                monitor = encoder.MultipartEncoderMonitor(m_encoder,
                                                          pg_callback
                                                          )
                label = fmfile['thefilename'] + ': '

                if pm.COMMANDLINE:
                    bar = ProgressBar(label=label,
                                      expected_size=fmfile['totalsize'])

                headers = {'Content-Type': m_encoder.content_type}

                res = self.session.post(url,
                                        params=fmfile,
                                        data=monitor,
                                        headers=headers)

                if res.status_code != 200:
                    hellraiser(res)

        #logger.info('\r')
        if auto_complete:
            return self.complete()

        return res