import random

def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """

    def f(x):
      return (x**4 - 3 * (x**3)) + 2
    def grad(x):
      return (4 * pow(x, 3) - 9 * pow(x, 2))

    best_x = start_x
    best_f = f(start_x)

    starting_points = [start_x] + [random.uniform(-5.0, 5.0) for i in range(10)] 

    for current_x in starting_points:
      x_old = current_x 
      lr = 0.001
      epsi = 0.000001

      while True:
        x_new = x_old - lr * grad(x_old)

        if abs(x_old - x_new) < epsi:
          x_old = x_new
          break

        x_old = x_new

      current_depth = f(x_old)

      if current_depth < best_f:
        best_f = current_depth
        best_x = x_old 

    return best_x
