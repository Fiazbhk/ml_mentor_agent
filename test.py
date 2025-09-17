import asyncio
from ml_solver import MLSolver

async def test_solver():
    s = MLSolver()
    cases = [
        "Explain bias-variance tradeoff",
        "Debug: training loss increases across epochs, here's a log: Loss: 0.8 -> 1.2 -> 1.5",
        "Create a one-day lesson plan for teaching attention",
        "Recommend model for small dataset of 500 labeled images"
    ]
    for c in cases:
        print("\n---\nInput:", c)
        out = await s.solve_problem(c)
        print("Output:\n", out)

if __name__ == '__main__':
    asyncio.run(test_solver())
