def hid(manufacturer: str, serial_number: str, model: str) -> str:
        """Computes the HID for the given properties of a device. The HID is suitable to use to an URI."""
        return Naming.url_word(manufacturer) + '-' + Naming.url_word(serial_number) + '-' + Naming.url_word(model)