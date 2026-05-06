def _show_outputs(self):
        """
        Print the terraform outputs.
        """
        outs = self._get_outputs()
        print("\n\n" + '=> Terraform Outputs:')
        for k in sorted(outs):
            print('%s = %s' % (k, outs[k]))