def init_device(self):
        """
        Initializes the device with the proper keymaps and name
        """
        try:
            product_id = int(self._send_command('_d2', 1))
        except ValueError:
            product_id = self._send_command('_d2', 1)

        if product_id == 0:
            self._impl = ResponseDevice(
                self.con,
                'Cedrus Lumina LP-400 Response Pad System',
                lumina_keymap)
        elif product_id == 1:
            self._impl = ResponseDevice(
                self.con,
                'Cedrus SV-1 Voice Key',
                None,
                'Voice Response')
        elif product_id == 2:
            model_id = int(self._send_command('_d3', 1))
            if model_id == 1:
                self._impl = ResponseDevice(
                    self.con,
                    'Cedrus RB-530',
                    rb_530_keymap)
            elif model_id == 2:
                self._impl = ResponseDevice(
                    self.con,
                    'Cedrus RB-730',
                    rb_730_keymap)
            elif model_id == 3:
                self._impl = ResponseDevice(
                    self.con,
                    'Cedrus RB-830',
                    rb_830_keymap)
            elif model_id == 4:
                self._impl = ResponseDevice(
                    self.con,
                    'Cedrus RB-834',
                    rb_834_keymap)
            else:
                raise XidError('Unknown RB Device')
        elif product_id == 4:
            self._impl = StimTracker(
                self.con,
                'Cedrus C-POD')
        elif product_id == b'S':
            self._impl = StimTracker(
                self.con,
                'Cedrus StimTracker')

        elif product_id == -99:
            raise XidError('Invalid XID device')