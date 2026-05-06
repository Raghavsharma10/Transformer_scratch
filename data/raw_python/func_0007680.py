def _apply_options(self, token):
        """Applies various filtering and processing options on token.

        Returns:
            The processed token. None if filtered.
        """
        # Apply work token filtering.
        if token.is_punct and self.remove_punct:
            return None
        if token.is_stop and self.remove_stop_words:
            return None
        if token.is_digit and self.remove_digits:
            return None
        if token.is_oov and self.exclude_oov:
            return None
        if token.pos_ in self.exclude_pos_tags:
            return None
        if token.ent_type_ in self.exclude_entities:
            return None

        # Lemmatized ones are already lowered.
        if self.lemmatize:
            return token.lemma_
        if self.lower:
            return token.lower_
        return token.orth_