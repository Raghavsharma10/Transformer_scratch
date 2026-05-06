def _gen_sentence(self, assetid_body_tuple):
        '''
        Takes an assetid_body_tuple and returns a Doc2Vec LabeledSentence 

        Args:
            assetid_body_tuple (tuple): (assetid, bodytext) pair 
        '''
        asset_id, body = assetid_body_tuple
        text = self._process(body)
        sentence = LabeledSentence(text, labels=['DOC_%s' % str(asset_id)])
        return sentence