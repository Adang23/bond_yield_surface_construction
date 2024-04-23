
def generate_constraints(initial_ratings):
    bounds = [(0,1) for _ in initial_ratings]

    index_length = len(initial_ratings)
    constraints = [
        {'type': 'ineq', 'fun': (lambda i: lambda r: r[i+1] - r[i] - 0.02) (i)}
        for i in range(index_length - 3)
    ]

    constraints.extend([
        {'type': 'ineq', 'fun': (lambda r: r[0] - 0.15)},
        {'type': 'ineq', 'fun': (lambda r: r[index_length -2] - r[index_length-3]) -0.2 },
        {'type': 'ineq', 'fun': (lambda r: r[index_length -1] - r[index_length-2]) -0.2 },
        {'type': 'eq',   'fun': (lambda r: r[index_length -1] - 1) }
    ])

    return bounds, constraints
