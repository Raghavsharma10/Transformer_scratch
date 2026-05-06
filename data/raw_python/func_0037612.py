def set_shipping(self, *args, **kwargs):
        ''' Define os atributos do frete

        Args:
            type (int): (opcional) Tipo de frete. Os valores válidos são: 1 para 'Encomenda normal (PAC).',
                2 para 'SEDEX' e 3 para 'Tipo de frete não especificado.'
            cost (float): (opcional) Valor total do frete. Deve ser maior que 0.00 e menor ou igual a 9999999.00.
            street (str): (opcional) Nome da rua do endereço de envio do produto
            address_number: (opcional) Número do endereço de envio do produto. 
            complement: (opcional) Complemento (bloco, apartamento, etc.) do endereço de envio do produto. 
            district: (opcional) Bairro do endereço de envio do produto.
            postal_code: (opcional) CEP do endereço de envio do produto.
            city: (opcional) Cidade do endereço de envio do produto.
            state: (opcional) Estado do endereço de envio do produto.
            country: (opcional) País do endereço de envio do produto. Apenas o valor 'BRA' é aceito.
        
        '''
        self.shipping = {}
        for arg, value in kwargs.iteritems():
            self.shipping[arg] = value
        shipping_schema(self.shipping)