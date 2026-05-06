def find_usbserial(vendor, product):
  """Find the tty device for a given usbserial devices identifiers.

  Args:
     vendor: (int) something like 0x0000
     product: (int) something like 0x0000

  Returns:
     String, like /dev/ttyACM0 or /dev/tty.usb...
  """
  if platform.system() == 'Linux':
    vendor, product = [('%04x' % (x)).strip() for x in (vendor, product)]
    return linux_find_usbserial(vendor, product)
  elif platform.system() == 'Darwin':
    return osx_find_usbserial(vendor, product)
  else:
    raise NotImplementedError('Cannot find serial ports on %s'
                              % platform.system())