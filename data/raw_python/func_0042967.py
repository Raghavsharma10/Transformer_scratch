def remove_device(self, path):
        "Remove a device from the daemon's internal search list."
        if self.__get_control_socket():
            self.sock.sendall("-%s\r\n\x00" % path)
            self.sock.recv(12)
            self.sock.close()