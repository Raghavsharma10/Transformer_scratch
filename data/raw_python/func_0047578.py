def pay_loan(self, loan_name):
        """additional when step for loan payments"""
        loan_payments = {
            'student': 600,
            'car': 200
        }
        payment_amount = loan_payments[loan_name]
        self.withdraw(payment_amount)