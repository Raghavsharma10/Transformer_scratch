def enrol(self, event):
        """A user tries to self-enrol with the enrolment form"""

        if self.config.allow_registration is False:
            self.log('Someone tried to register although enrolment is closed.')
            return

        self.log('Client trying to register a new account:', event, pretty=True)
        # self.log(event.data, pretty=True)

        uuid = event.client.uuid

        if uuid in self.captchas and event.data.get('captcha', None) == self.captchas[uuid]['text']:
            self.log('Captcha solved!')
        else:
            self.log('Captcha failed!')
            self._fail(event, _('You did not solve the captcha correctly.', event))
            self._generate_captcha(event)

            return

        mail = event.data.get('mail', None)
        if mail is None:
            self._fail(event, _('You have to supply all required fields.', event))
            return
        elif not validate_email(mail):
            self._fail(event, _('The supplied email address seems invalid', event))
            return

        if objectmodels['user'].count({'mail': mail}) > 0:
            self._fail(event, _('Your mail address cannot be used.', event))
            return

        password = event.data.get('password', None)
        if password is None or len(password) < 5:
            self._fail(event, _('Your password is not long enough.', event))
            return

        username = event.data.get('username', None)
        if username is None or len(username) < 1:
            self._fail(event, _('Your username is not long enough.', event))
            return
        elif (objectmodels['user'].count({'name': username}) > 0) or \
            (objectmodels['enrollment'].count({'name': username}) > 0):
            self._fail(event, _('The username you supplied is not available.', event))
            return

        self.log('Provided data is good to enrol.')
        if self.config.no_verify:
            self._create_user(username, password, mail, 'Enrolled', uuid)
        else:
            self._invite(username, 'Enrolled', mail, uuid, event, password)