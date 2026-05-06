def _label_names_correct(self, labels):
        """Raise exception (ValueError) if labels not correct"""

        for k, v in labels.items():
            # Check reserved labels
            if k in RESTRICTED_LABELS_NAMES:
                raise ValueError("Labels not correct")

            # Check prefixes
            if any(k.startswith(i) for i in RESTRICTED_LABELS_PREFIXES):
                raise ValueError("Labels not correct")

        return True