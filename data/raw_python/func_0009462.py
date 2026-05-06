def setCamera(self, camera_name, bit_depth=16):
        '''
        Args:
            camera_name (str): Name of the camera
            bit_depth (int): depth (bit) of the camera sensor
        '''
        self.coeffs['name'] = camera_name
        self.coeffs['depth'] = bit_depth