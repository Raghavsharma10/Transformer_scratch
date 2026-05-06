def set_manip(self, manipulatable):
        """stub"""
        if not isinstance(manipulatable, DataInputStream):
            raise InvalidArgument('Manipulatable object be an ' +
                                  'osid.transport.DataInputStream object')
        self.add_file(manipulatable,
                      label='manip',
                      asset_type=MANIP_ASSET_TYPE,
                      asset_content_type=MANIP_ASSET_CONTENT_TYPE)