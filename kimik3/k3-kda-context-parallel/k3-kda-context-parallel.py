import torch


def kda_context_parallel(
    transition_matrices: torch.Tensor,
    local_contributions: torch.Tensor,
    initial_state: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing incoming states, outgoing states, and the final global state.
    """
    segments: list[torch.Tensor] = []
    segments.append(initial_state)

    for transition_matrix, local_contribution in zip(
        transition_matrices, local_contributions
    ):
        incoming_state = segments[-1]  # always use the previous state

        outgoing_state = (
            torch.matmul(transition_matrix, incoming_state) + local_contribution
        )
        segments.append(outgoing_state)

    l = len(segments)
    d = {
        "incoming_states": torch.stack(segments[0 : l - 1]),
        "outgoing_states": torch.stack(segments[1:l]),
        "final_state": segments[l - 1],
    }
    return d
