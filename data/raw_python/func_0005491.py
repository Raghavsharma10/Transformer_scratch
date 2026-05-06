def create(self, items=[], taxes=[], custom_data=[]):
        """Adds the items to the invoice

        Format of 'items':
        [
         InvoiceItem(
             name="VIP Ticket",
             quantity= 2,
             unit_price= "3500",
             total_price= "7000",
             description= "VIP Tickets for the Party"
          }
        ,...
        ]
        """
        self.add_items(items)
        self.add_taxes(taxes)
        self.add_custom_data(custom_data)
        return self._process('checkout-invoice/create', self._prepare_data)