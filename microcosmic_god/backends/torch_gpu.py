from __future__ import annotations

from collections import defaultdict

from .contracts import BrainLearningCase
from microcosmic_god.brain import AUXILIARY_PREDICTION_HEADS, PREDICTION_HEADS, TinyBrain


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _ensure_brain_state(brain: TinyBrain) -> None:
    if len(brain.hidden) != brain.hidden_size:
        brain.hidden = [0.0 for _ in range(brain.hidden_size)]
    if len(brain.last_outputs) != brain.output_size:
        brain.last_outputs = [0.0 for _ in range(brain.output_size)]
    if len(brain.last_inputs) != brain.input_size:
        brain.last_inputs = [0.0 for _ in range(brain.input_size)]
    if len(brain.input_trace) != brain.input_size:
        brain.input_trace = [0.0 for _ in range(brain.input_size)]
    if len(brain.hidden_trace) != brain.hidden_size:
        brain.hidden_trace = [0.0 for _ in range(brain.hidden_size)]


class TorchBrainRuntime:
    name = "torch"

    def __init__(self, device: str = "auto"):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for --backend torch") from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device")
        self.torch = torch
        self.device = device
        self.dtype = torch.float32

    def forward_many(self, brains: list[TinyBrain], observations: list[list[float]]) -> list[list[float]]:
        if len(brains) != len(observations):
            raise ValueError("brains and observations must have matching lengths")
        outputs: list[list[float] | None] = [None for _ in brains]
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, brain in enumerate(brains):
            _ensure_brain_state(brain)
            groups[(brain.input_size, brain.hidden_size, brain.output_size)].append(index)

        for (input_size, hidden_size, output_size), indexes in groups.items():
            batch = len(indexes)
            torch = self.torch
            inv = 1.0 / (max(1, input_size) ** 0.5)
            inv_h = 1.0 / (max(1, hidden_size) ** 0.5)
            x = torch.tensor([observations[i] for i in indexes], dtype=self.dtype, device=self.device)
            hidden = torch.tensor([brains[i].hidden for i in indexes], dtype=self.dtype, device=self.device)
            weights_in = torch.tensor(
                [brains[i].weights_in for i in indexes],
                dtype=self.dtype,
                device=self.device,
            ).view(batch, hidden_size, input_size)
            weights_out = torch.tensor(
                [brains[i].weights_out for i in indexes],
                dtype=self.dtype,
                device=self.device,
            ).view(batch, output_size, hidden_size)
            bias_h = torch.tensor([brains[i].bias_h for i in indexes], dtype=self.dtype, device=self.device)
            bias_o = torch.tensor([brains[i].bias_o for i in indexes], dtype=self.dtype, device=self.device)
            input_trace = torch.tensor([brains[i].input_trace for i in indexes], dtype=self.dtype, device=self.device)
            hidden_trace = torch.tensor([brains[i].hidden_trace for i in indexes], dtype=self.dtype, device=self.device)

            input_trace = input_trace * 0.92 + x * 0.08
            new_hidden = torch.tanh(bias_h + hidden * 0.62 + torch.einsum("bhi,bi->bh", weights_in, x) * inv)
            hidden_trace = hidden_trace * 0.90 + new_hidden * 0.10
            out = bias_o + torch.einsum("boh,bh->bo", weights_out, new_hidden) * inv_h

            x_rows = x.detach().cpu().tolist()
            input_trace_rows = input_trace.detach().cpu().tolist()
            hidden_rows = new_hidden.detach().cpu().tolist()
            hidden_trace_rows = hidden_trace.detach().cpu().tolist()
            output_rows = out.detach().cpu().tolist()
            for local, brain_index in enumerate(indexes):
                brain = brains[brain_index]
                brain.last_inputs = [float(value) for value in x_rows[local]]
                brain.input_trace = [float(value) for value in input_trace_rows[local]]
                brain.hidden = [float(value) for value in hidden_rows[local]]
                brain.hidden_trace = [float(value) for value in hidden_trace_rows[local]]
                brain.last_outputs = [float(value) for value in output_rows[local]]
                outputs[brain_index] = brain.last_outputs

        return [row if row is not None else [] for row in outputs]

    def learn_many(self, cases: list[BrainLearningCase]) -> list[float]:
        errors: list[float] = [0.0 for _ in cases]
        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, case in enumerate(cases):
            brain = case.brain
            _ensure_brain_state(brain)
            groups[(brain.input_size, brain.hidden_size, brain.output_size)].append(index)

        for (input_size, hidden_size, output_size), indexes in groups.items():
            self._learn_group(cases, indexes, input_size, hidden_size, output_size, errors)
        return errors

    def _learn_group(
        self,
        cases: list[BrainLearningCase],
        indexes: list[int],
        input_size: int,
        hidden_size: int,
        output_size: int,
        errors_out: list[float],
    ) -> None:
        torch = self.torch
        batch = len(indexes)
        inv_h = 1.0 / (max(1, hidden_size) ** 0.5)
        brains = [cases[i].brain for i in indexes]

        hidden = torch.tensor([brain.hidden for brain in brains], dtype=self.dtype, device=self.device)
        hidden_trace = torch.tensor([brain.hidden_trace for brain in brains], dtype=self.dtype, device=self.device)
        input_trace = torch.tensor([brain.input_trace for brain in brains], dtype=self.dtype, device=self.device)
        weights_in = torch.tensor([brain.weights_in for brain in brains], dtype=self.dtype, device=self.device).view(batch, hidden_size, input_size)
        weights_out = torch.tensor([brain.weights_out for brain in brains], dtype=self.dtype, device=self.device).view(batch, output_size, hidden_size)
        bias_o = torch.tensor([brain.bias_o for brain in brains], dtype=self.dtype, device=self.device)
        prediction_weights = torch.tensor([brain.prediction_weights for brain in brains], dtype=self.dtype, device=self.device)
        auxiliary_weights = {
            head: torch.tensor(
                [brain.auxiliary_prediction_weights.get(head, [0.0 for _ in range(hidden_size)]) for brain in brains],
                dtype=self.dtype,
                device=self.device,
            )
            for head in AUXILIARY_PREDICTION_HEADS
        }

        action_index = torch.tensor([cases[i].action_index for i in indexes], dtype=torch.long, device=self.device)
        valence = torch.tensor([_clip(cases[i].valence, -2.0, 2.0) for i in indexes], dtype=self.dtype, device=self.device)
        learning_rate = torch.tensor([_clip(cases[i].learning_rate, 0.0, 0.25) for i in indexes], dtype=self.dtype, device=self.device)
        plasticity = torch.tensor([_clip(cases[i].plasticity, 0.0, 1.0) for i in indexes], dtype=self.dtype, device=self.device)
        prediction_weight = torch.tensor([_clip(cases[i].prediction_weight, 0.0, 1.0) for i in indexes], dtype=self.dtype, device=self.device)
        lr = learning_rate * plasticity
        pred_lr = lr * prediction_weight * 0.025

        target = torch.zeros((batch, len(PREDICTION_HEADS)), dtype=self.dtype, device=self.device)
        target_mask = torch.zeros((batch, len(PREDICTION_HEADS)), dtype=self.dtype, device=self.device)
        for row, case_index in enumerate(indexes):
            case = cases[case_index]
            targets: dict[str, float] = {"energy": case.energy_delta}
            targets.update(case.outcome_targets)
            for head_index, head in enumerate(PREDICTION_HEADS):
                if head in targets:
                    target[row, head_index] = _clip(targets[head], -2.0, 2.0)
                    target_mask[row, head_index] = 1.0

        prediction_columns = [torch.sum(prediction_weights * hidden, dim=1) * inv_h]
        for head in AUXILIARY_PREDICTION_HEADS:
            prediction_columns.append(torch.sum(auxiliary_weights[head] * hidden, dim=1) * inv_h)
        predictions = torch.stack(prediction_columns, dim=1)
        prediction_errors = torch.clamp(target - predictions, -2.0, 2.0) * target_mask

        energy_error = prediction_errors[:, 0]
        prediction_weights = torch.clamp(prediction_weights + pred_lr[:, None] * energy_error[:, None] * hidden, -4.0, 4.0)
        for head_index, head in enumerate(AUXILIARY_PREDICTION_HEADS, start=1):
            error = prediction_errors[:, head_index]
            mask = target_mask[:, head_index]
            auxiliary_weights[head] = torch.clamp(
                auxiliary_weights[head] + (pred_lr * mask)[:, None] * error[:, None] * hidden,
                -4.0,
                4.0,
            )

        batch_index = torch.arange(batch, device=self.device)
        weights_out[batch_index, action_index, :] = torch.clamp(
            weights_out[batch_index, action_index, :] + (lr * valence * 0.035)[:, None] * hidden_trace,
            -4.0,
            4.0,
        )
        bias_o[batch_index, action_index] = torch.clamp(
            bias_o[batch_index, action_index] + lr * valence * 0.015,
            -4.0,
            4.0,
        )

        representation_lr = lr * (0.15 + prediction_weight * 0.35) * 0.010
        surprise_count = torch.clamp(torch.sum(target_mask, dim=1), min=1.0)
        surprise = torch.sum(torch.abs(prediction_errors), dim=1) / surprise_count
        modulation = torch.clamp(
            valence * 0.55 + energy_error * prediction_weight * 0.35 + surprise * prediction_weight * 0.10,
            -2.0,
            2.0,
        )
        hidden_gate = torch.clamp(hidden_trace, -1.0, 1.0)
        hidden_mask = (torch.abs(hidden_trace) >= 0.015).to(self.dtype)
        input_mask = (torch.abs(input_trace) >= 0.010).to(self.dtype)
        delta_in = (
            (representation_lr * modulation)[:, None, None]
            * hidden_gate[:, :, None]
            * input_trace[:, None, :]
            * hidden_mask[:, :, None]
            * input_mask[:, None, :]
        )
        weights_in = torch.clamp(weights_in + delta_in, -4.0, 4.0)

        weights_in_rows = weights_in.reshape(batch, hidden_size * input_size).detach().cpu().tolist()
        weights_out_rows = weights_out.reshape(batch, output_size * hidden_size).detach().cpu().tolist()
        bias_o_rows = bias_o.detach().cpu().tolist()
        prediction_rows = prediction_weights.detach().cpu().tolist()
        auxiliary_rows = {head: weights.detach().cpu().tolist() for head, weights in auxiliary_weights.items()}
        error_rows = prediction_errors.detach().cpu().tolist()

        for local, case_index in enumerate(indexes):
            brain = cases[case_index].brain
            brain.weights_in = [float(value) for value in weights_in_rows[local]]
            brain.weights_out = [float(value) for value in weights_out_rows[local]]
            brain.bias_o = [float(value) for value in bias_o_rows[local]]
            brain.prediction_weights = [float(value) for value in prediction_rows[local]]
            for head in AUXILIARY_PREDICTION_HEADS:
                brain.auxiliary_prediction_weights[head] = [float(value) for value in auxiliary_rows[head][local]]
            brain.last_prediction_errors = {
                head: float(error_rows[local][head_index])
                for head_index, head in enumerate(PREDICTION_HEADS)
            }
            errors_out[case_index] = brain.last_prediction_errors["energy"]
