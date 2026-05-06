def build_emv_data(self):
        """
        TODO:
        95    TVR
        82    app_int_prof 
        """
        emv_data = ''
        emv_data += self.TLV.build({'82': self._get_app_interchange_profile()})
        emv_data += self.TLV.build({'9A': get_date()})
        emv_data += self.TLV.build({'95': self.term.get_tvr()})
        emv_data += self.TLV.build({'9F10': self.card.get_iss_application_data()})
        emv_data += self.TLV.build({'9F26': self.card.get_application_cryptogram()})
        emv_data += self.TLV.build({'9F36': self.card.get_transaction_counter()})
        emv_data += self.TLV.build({'9F37': self.term.get_unpredno()})
        emv_data += self.TLV.build({'9F1A': self.term.get_country_code()})
        return emv_data