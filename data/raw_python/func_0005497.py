def __encode_items(self, items):
        """Encodes the InvoiceItems into a JSON serializable format

        items = [('item_1',InvoiceItem(name='VIP Ticket', quantity=2,
                             unit_price='3500', total_price='7000',
                             description='VIP Tickets for party')),...]
        """
        xs = [item._asdict() for (_key, item) in items.items()]
        return list(map(lambda x: dict(zip(x.keys(), x.values())), xs))