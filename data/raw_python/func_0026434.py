def captcha_transmit(self, captcha, uuid):
        """Delayed transmission of a requested captcha"""

        self.log('Transmitting captcha')

        response = {
            'component': 'hfos.enrol.enrolmanager',
            'action': 'captcha',
            'data': b64encode(captcha['image'].getvalue()).decode('utf-8')
        }
        self.fire(send(uuid, response))