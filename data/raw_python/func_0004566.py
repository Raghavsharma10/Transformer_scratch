def get_available_ip6_for_vip(self, id_evip, name):
        """
        Get and save a available IP in the network ipv6 for vip request

        :param id_evip: Vip environment identifier. Integer value and greater than zero.
        :param name: Ip description

        :return: Dictionary with the following structure:

        ::

            {'ip': {'bloco1':<bloco1>,
            'bloco2':<bloco2>,
            'bloco3':<bloco3>,
            'bloco4':<bloco4>,
            'bloco5':<bloco5>,
            'bloco6':<bloco6>,
            'bloco7':<bloco7>,
            'bloco8':<bloco8>,
            'id':<id>,
            'networkipv6':<networkipv6>,
            'description':<description>}}

        :raise IpNotAvailableError: Network dont have available IP for vip environment.
        :raise EnvironmentVipNotFoundError: Vip environment not registered.
        :raise UserNotAuthorizedError: User dont have permission to perform operation.
        :raise InvalidParameterError: Vip environment identifier is none or invalid.
        :raise XMLError: Networkapi failed to generate the XML response.
        :raise DataBaseError: Networkapi failed to access the database.

        """

        if not is_valid_int_param(id_evip):
            raise InvalidParameterError(
                u'Vip environment identifier is invalid or was not informed.')

        url = 'ip/availableip6/vip/' + str(id_evip) + "/"

        ip_map = dict()
        ip_map['id_evip'] = id_evip
        ip_map['name'] = name

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)