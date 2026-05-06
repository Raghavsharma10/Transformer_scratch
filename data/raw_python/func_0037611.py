def set_client(self, *args, **kwargs):
        ''' Se você possui informações cadastradas sobre o comprador você pode utilizar
        este método para enviar estas informações para o PagSeguro. É uma boa prática pois
        evita que seu cliente tenha que preencher estas informações novamente na página
        do PagSeguro.

        Args:
            name (str): (opcional) Nome do cliente
            email (str): (opcional) Email do cliente
            phone_area_code (str): (opcional) Código de área do telefone do cliente. Um número com 2 digitos.
            phone_number (str): (opcional) O número de telefone do cliente.
            cpf: (str): (opcional) Número do cpf do comprador
            born_date: (date): Data de nascimento no formato dd/MM/yyyy

        Exemplo:
            >>> from pagseguro import Payment
            >>> from pagseguro import local_settings
            >>> payment = Payment(email=local_settings.PAGSEGURO_ACCOUNT_EMAIL, token=local_settings.PAGSEGURO_TOKEN, sandbox=True)
            >>> payment.set_client(name=u'Adam  Yauch', phone_area_code=11)
        '''
        self.client = {}      
        for arg, value in kwargs.iteritems():
            if value:
                self.client[arg] = value
        client_schema(self.client)