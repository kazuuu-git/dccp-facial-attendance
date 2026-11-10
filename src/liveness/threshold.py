from math import isclose, isfinite


def evaluate_liveness(model_scores, *, threshold):
    """
    Evaluate one or more three-class softmax score vectors.

    Input example:
        [[0.05, 0.90, 0.05], [0.10, 0.85, 0.05]]

    Class index 1 is LIVE.
    The threshold must be explicitly supplied.
    """

    threshold = float(threshold)

    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be a finite number from 0 to 1.")

    def invalid(reason):
        return {
            "status": "INVALID",
            "passed": False,
            "live_score": None,
            "average_scores": None,
            "threshold": threshold,
            "reason": reason,
        }

    try:
        rows = [
            [float(value) for value in scores]
            for scores in model_scores
        ]
    except (TypeError, ValueError, OverflowError):
        return invalid("Scores must contain numeric score vectors.")

    if not rows:
        return invalid("No model scores supplied.")

    for row in rows:
        if len(row) != 3:
            return invalid("Each model must supply exactly three scores.")

        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in row):
            return invalid("Scores must be finite values from 0 to 1.")

        if not isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-5):
            return invalid("Each model's softmax scores must sum to 1.")

    average_scores = [
        sum(row[index] for row in rows) / len(rows)
        for index in range(3)
    ]

    live_score = average_scores[1]

    # A tie with a spoof class does not qualify as a LIVE win.
    live_wins = (
        live_score > average_scores[0]
        and live_score > average_scores[2]
    )

    passed = live_wins and live_score >= threshold

    if not live_wins:
        reason = "LIVE is not the unique highest-scoring class."
    elif not passed:
        reason = "Live score is below the threshold."
    else:
        reason = "Live score meets the acceptance rule."

    return {
        "status": "PASS" if passed else "REJECT",
        "passed": passed,
        "live_score": live_score,
        "average_scores": average_scores,
        "threshold": threshold,
        "reason": reason,
    }