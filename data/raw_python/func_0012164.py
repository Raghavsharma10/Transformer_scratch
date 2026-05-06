def validate(self, tracking_number):
        "Return True if this is a valid UPS tracking number."
        tracking_num = tracking_number[2:-1]
        odd_total = 0
        even_total = 0

        for ii, digit in enumerate(tracking_num.upper()):
            try:
                value = int(digit)
            except ValueError:
                value = int((ord(digit) - 63) % 10)
            if (ii + 1) % 2:
                odd_total += value
            else:
                even_total += value

        total = odd_total + even_total * 2
        check = ((total - (total % 10) + 10) - total) % 10
        return (check == int(tracking_number[-1:]))