def get_nth_bit(number, n):
    mask = 2**n
    return (number & mask)//mask

def invert_nth_bit(number, n):
    bit = get_nth_bit(number, n)
    return number + (-1)**bit*2**n

