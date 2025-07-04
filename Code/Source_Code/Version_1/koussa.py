import math

def get_grid_dimensions(n):
    n = min(n, 9)  # Cap at 9
    cols = min(3, n)
    rows = math.ceil(n / cols)
    return rows, cols

# Test for 1 to 20
for i in range(1, 21):
    rows, cols = get_grid_dimensions(i)
    print(f"{i} graph(s): {rows} row(s) x {cols} column(s)")