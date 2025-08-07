"""
Demonstrate line-by-line performance profiling of a Python function using `line_profiler`.
Expected Output: Profiling report indicating which lines are performance bottlenecks.
"""

import time

def slow_function():
    total = 0
    for i in range(10000):
        for j in range(100):
            total += i * j
    return total

if __name__ == "__main__":
    slow_function()
