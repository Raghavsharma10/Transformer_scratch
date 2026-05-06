def insert_rack(
            self,
            number,
            name,
            mac_address_sw1,
            mac_address_sw2,
            mac_address_ilo,
            id_sw1,
            id_sw2,
            id_ilo):
        """Create new Rack
        :param number: Number of Rack
        :return: Following dictionary:
        ::
          {'rack': {'id': < id_rack >,
          'num_rack': < num_rack >,
          'name_rack': < name_rack >,
          'mac_sw1': < mac_sw1 >,
          'mac_sw2': < mac_sw2 >,
          'mac_ilo': < mac_ilo >,
          'id_sw1': < id_sw1 >,
          'id_sw2': < id_sw2 >,
          'id_ilo': < id_ilo >, } }
        :raise RacksError: Rack already registered with informed.
        :raise NumeroRackDuplicadoError: There is already a registered Rack with the value of number.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(number):
            raise InvalidParameterError(u'Rack number is none or invalid')

        rack_map = dict()
        rack_map['number'] = number
        rack_map['name'] = name
        rack_map['mac_address_sw1'] = mac_address_sw1
        rack_map['mac_address_sw2'] = mac_address_sw2
        rack_map['mac_address_ilo'] = mac_address_ilo
        rack_map['id_sw1'] = id_sw1
        rack_map['id_sw2'] = id_sw2
        rack_map['id_ilo'] = id_ilo

        code, xml = self.submit({'rack': rack_map}, 'POST', 'rack/insert/')

        return self.response(code, xml)