"""Решения задач по основам Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    """Считает латинские гласные в строке без учёта регистра."""
    vowels = set("aeiou")
    return sum(char.lower() in vowels for char in data.value)


def has_unique_characters(data: TextInput) -> bool:
    """Проверяет, встречается ли каждый символ в строке не более одного раза."""
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    """Считает единичные биты в двоичной записи положительного числа."""
    return data.value.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    """Считает количество перемножений цифр до получения однозначного числа."""
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
    """Вычисляет среднюю квадратичную ошибку двух векторов одинаковой длины."""
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("Vectors must have the same length")
    if len(predicted) == 0:
        raise ValueError("MSE is undefined for empty vectors")
    return sum((a - b) ** 2 for a, b in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    """Возвращает разложение числа на простые множители в порядке возрастания."""
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
    # У числа 1 нет простых множителей, поэтому результат — пустое произведение.
    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    """Возвращает k, если n = 1² + ... + k²; в противном случае сообщает об ошибке."""
    cube_count = data.value
    total = 0
    level = 0
    while total < cube_count:
        level += 1
        total += level * level
    return level if total == cube_count else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    """Сравнивает суммы цифр по обе стороны середины числа, не считая середину."""
    digits = str(data.value)
    middle = len(digits) // 2
    if len(digits) % 2:
        left, right = digits[:middle], digits[middle + 1 :]
    else:
        left, right = digits[: middle - 1], digits[middle + 1 :]
    return sum(map(int, left)) == sum(map(int, right))
