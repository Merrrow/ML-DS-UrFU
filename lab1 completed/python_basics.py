"""Решения задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    """Return the number of Latin vowels in a string, ignoring case."""
    vowels = set("aeiou")
    return sum(char.lower() in vowels for char in data.value)


def has_unique_characters(data: TextInput) -> bool:
    """Return whether every character appears at most once."""
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    """Count the set bits in the binary representation of a positive integer."""
    return data.value.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    """Count digit-product steps required to reach a single digit."""
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    """Return the mean squared error between two equal-length vectors."""
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")
    if len(predicted) == 0:
        raise ValueError("MSE is undefined for empty vectors")
    return sum((a - b) ** 2 for a, b in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    """Format a positive integer's prime factorization in ascending order."""
    number = data.value
    factors: list[str] = []
    divisor = 2
    while divisor * divisor <= number:
        exponent = 0
        while number % divisor == 0:
            number //= divisor
            exponent += 1
        if exponent:
            factors.append(f"({divisor}**{exponent})" if exponent > 1 else f"({divisor})")
        divisor = 3 if divisor == 2 else divisor + 2
    if number > 1:
        factors.append(f"({number})")
    # 1 has no prime factors, so its factorization is the empty product.
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    """Return k if n = 1² + ... + k², or the failure message otherwise."""
    cube_count = data.value
    total = 0
    level = 0
    while total < cube_count:
        level += 1
        total += level * level
    return level if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    """Compare digit sums on both sides, excluding the middle digit(s)."""
    digits = str(data.value)
    middle = len(digits) // 2
    if len(digits) % 2:
        left, right = digits[:middle], digits[middle + 1 :]
    else:
        left, right = digits[: middle - 1], digits[middle + 1 :]
    return sum(map(int, left)) == sum(map(int, right))
