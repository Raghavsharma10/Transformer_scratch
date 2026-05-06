def makeProductPicker(self):
        """
        Make a LiveForm with radio buttons for each Product in the store.
        """
        productPicker = liveform.LiveForm(
            self.coerceProduct,
            [liveform.Parameter(
              str(id(product)),
              liveform.FORM_INPUT,
              liveform.LiveForm(
              lambda selectedProduct, product=product: selectedProduct and product,
              [liveform.Parameter(
                'selectedProduct',
                liveform.RADIO_INPUT,
                bool,
                repr(product))]
              ))
              for product
              in self.original.store.parent.query(Product)],
            u"Product to Install")
        return productPicker