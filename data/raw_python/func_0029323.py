def get_description(self):
        """
        Get transaction description (for logging purposes)
        """
        if self.card:
            card_description = self.card.get_description()
        else:
            card_description = 'Cardless'

        if card_description:
            card_description += ' | '

        return card_description + self.description if self.description else card_description + self.type + ' ' + str(self.IsoMessage.FieldData(11))