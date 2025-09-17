import re
import asyncio
from typing import List


class MLSolver:
    """A lightweight, educational ML solver used as the agent's domain-specific logic.

    Capabilities implemented:
    - Explain ML concepts step-by-step
    - Diagnose training/debug problems from small code snippets / logs
    - Recommend model selection and hyperparameters
    - Generate short lesson plans and exercises
    """

    def __init__(self):
        self.knowledge = {
            "concepts": {
                "bias_variance": "Bias-variance tradeoff describes the tension between model complexity and generalization...",
                "regularization": "Regularization techniques (L1, L2, dropout) help prevent overfitting by...",
                "transformer_overview": "Transformers use self-attention to compute contextual representations..."
            }
        }

    async def solve_problem(self, problem: str) -> str:
        problem_lower = problem.lower()
        if any(w in problem_lower for w in ["explain", "what is", "define"]):
            return self._explain_concept(problem)
        if any(w in problem_lower for w in ["debug", "error", "loss", "nan", "diverge"]):
            return self._debug_problem(problem)
        if any(w in problem_lower for w in ["lesson", "plan", "curriculum", "teach"]):
            return self._lesson_plan(problem)
        if any(w in problem_lower for w in ["recommend", "hyper", "architecture", "model"]):
            return self._recommend(problem)
        # fallback
        return self._general_guidance(problem)

    def _explain_concept(self, text: str) -> str:
        for key, desc in self.knowledge["concepts"].items():
            if key.replace("_", " ") in text.lower():
                return (
                    f"**Explanation: {key.replace('_', ' ').title()}**\n\n"
                    f"{desc}\n\n"
                    f"Example and simple intuition: ..."
                )
        return (
            "I can explain core ML concepts (bias-variance, regularization, transformers, overfitting). "
            "Which one would you like?"
        )

    def _debug_problem(self, text: str) -> str:
        # Attempt to extract simple patterns: exploding loss, learning rate issues, data leakage
        tips: List[str] = []
        text_lower = text.lower()

        if "nan" in text_lower:
            tips.append("Check for numerical instability: gradient explosion, division by zero, or invalid values in input tensors.")
        if "loss increases" in text_lower or "diverge" in text_lower:
            tips.append("Try decreasing the learning rate, check for label leakage, and verify that your model's initialization is reasonable.")
        if "overfit" in text_lower or "overfitting" in text_lower:
            tips.append("Add regularization (weight decay), use dropout, or collect more data / simpler model.")

        # Quick code-snippet heuristic: detect missing optimizer.step() or zeroing grads
        if re.search(r"optimizer\.step\(|optimizer\.zero_grad\(|loss\.backward\(", text):
            tips.append("Verify training loop contains loss.backward(), optimizer.step(), and optimizer.zero_grad()/optimizer.clear_grad().")

        if not tips:
            tips.append("I couldn't find an obvious heuristic from the short snippet. Share the training loop or a small reproducible example and I'll analyze it.")

        return "\n".join(["**Debugging Tips**:"] + [f"- {t}" for t in tips])

    def _lesson_plan(self, text: str) -> str:
        # Determine target audience and duration heuristically
        duration = "1 week"
        if "one hour" in text or "30 minutes" in text:
            duration = "1 hour"
        elif "semester" in text or "course" in text:
            duration = "12 weeks"

        return (
            f"**Lesson Plan ({duration}) — Teaching Transformers**\n\n"
            "Day 1: Attention mechanism + motivation.\n"
            "Day 2: Scaled dot-product attention + multi-head attention.\n"
            "Day 3: Encoder-decoder architecture and positional encodings.\n"
            "Day 4: Hands-on: build a small transformer in PyTorch/TensorFlow.\n"
            "Day 5: Fine-tuning, transfer learning, and evaluation metrics.\n\n"
            "**Exercises:** Implement attention from scratch; fine-tune a small pretrained model on a tiny dataset."
        )

    def _recommend(self, text: str) -> str:
        text_lower = text.lower()
        if "small dataset" in text_lower or "few" in text_lower:
            return (
                "For small datasets: prefer simpler models (logistic regression, small MLP), cross-validation, "
                "data augmentation, and Bayesian or regularized approaches. Consider transfer learning if using images or NLP."
            )
        return (
            "For model selection: consider your dataset size, label noise, compute budget, and performance targets. "
            "Start simple, run strong baselines, then iterate."
        )

    def _general_guidance(self, text: str) -> str:
        return (
            "I can help with: concept explanations, debugging training loops, model selection, hyperparameter suggestions, and lesson plans. "
            "Provide a short code snippet, dataset description, or specific learning objective and I will respond with a step-by-step answer."
        )
