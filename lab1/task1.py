def AND(x1, x2):
    return 1 if x1 + x2 >= 2 else 0

def OR(x1, x2):
    return 1 if x1 + x2 >= 1 else 0

def XOR(x1, x2):
    y1 = OR(x1, x2)
    y2 = AND(x1, x2)
    return AND(y1, 1 - y2)

# Тестування
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"XOR({x1},{x2}) = {XOR(x1, x2)}")
