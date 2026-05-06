def open(self):
        """ Search device on USB tree and set is as escpos device """
        
        self.device = usb.core.find(idVendor=self.idVendor, idProduct=self.idProduct)
        if self.device is None:
            raise NoDeviceError()
        try:
            if self.device.is_kernel_driver_active(self.interface):
                self.device.detach_kernel_driver(self.interface) 
            self.device.set_configuration()
            usb.util.claim_interface(self.device, self.interface)
        except usb.core.USBError as e:
            raise HandleDeviceError(e)