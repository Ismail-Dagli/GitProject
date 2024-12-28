if __name__ == '__main__':
    def calculate_square_root(x):
        """
        Calculate the square root of a number using the exponentiation operator.

        This function calculates the square root of a given number by raising it
        to the power of 0.5. The input number must be non-negative, as this
        method does not support complex numbers.

        :param x: A non-negative number for which the square root will be calculated.
        :type x: float
        :return: The square root of the input number.
        :rtype: float
        """
        return x ** 0.5


    result = calculate_square_root(25)
    print(result)
