import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from typing import List, Dict, Tuple, Any, Union, Optional

# Constants
TERM_NAMES = [
    "Monod Growth",
    "Hill Kinetics Growth",
    "Substrate Inhibition Factor",
    "Product Inhibition Factor (Competitive)",
    "Non-Competitive Product Inhibition",
    "Competitive Inhibition Factor",
    "Double Substrate Limited Factor",
    "Substrate Threshold Activation",
    "Inhibitor Saturation",
]


def candidate_models(
    y: Union[List[float], np.ndarray],
    t: Union[float, np.ndarray],
    params: List[float],
    mask: List[int],
) -> List[float]:
    """
    Calculate derivatives based on selected growth mechanisms.

    Parameters
    ----------
    y : Union[List[float], np.ndarray]
        State variables [X, S, P, I] representing biomass, substrate,
        product, and inhibitor concentrations.
    t : Union[float, np.ndarray]
        Time point(s) at which to evaluate the model.
    params : List[float]
        Model parameters [mu_max, Ks, Ki, Yxs, Kp].
    mask : List[int]
        Binary mask indicating which growth terms to include.

    Returns
    -------
    List[float]
        Derivatives [dX/dt, dS/dt, dP/dt, dI/dt] at the given time point.
    """
    X, S, P, I = y
    mu_max, Ks, Kp, Yxs, Ki = params

    growth = mu_max * X
    # Growth terms
    # Monod Growth
    if mask[0]:
        growth *= S / (Ks + S)

    # Hill Kinetics Growth
    if mask[1]:
        n = 2
        growth *= S**n / (Ks**n + S**n)

    # Substrate Inhibition Factor
    if mask[2]:
        growth *= 1 / (1 + S / Ki)

    # Product Inhibition Factor (Competitive)
    if mask[3]:
        growth *= (
            (Ks + S) / (S + Ks + (Ks * P / Kp))
        )  # This term is modified so that is can be combined with mask 0 or 1 for a correct inhibition

    # Non-Competitive Product Inhibition
    if mask[4]:
        growth *= 1 / (1 + P / Kp)

    # Competitive Inhibition Factor
    if mask[5]:
        growth *= 1 / (1 + I / Ki)

    # Double Substrate Limited Factor (Inhibitor is a second substrate in case of an inhibitor)
    if mask[6]:
        growth *= I / (Ki + I)

    # Substrate Threshold Activation
    if mask[7]:
        S_threshold = 0.5
        growth *= (S - S_threshold) / (Ks + (S - S_threshold)) if S > S_threshold else 0

    # Inhibitor Saturation
    if mask[8]:
        growth *= 1 / (1 + (P / (P + Ki)))

    # Calculate derivatives
    dX = growth
    dS = -(growth / Yxs)
    dP = 0.3 * growth
    dI = -0.1 * I

    return [dX, dS, dP, dI]


def generate_training_data(
    initial_conditions: Union[List[float], List[List[float]]],
    true_model: callable,
    n_timepoints: int = 20,
    noise_level: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Generate training data with noise.

    Parameters
    ----------
    initial_conditions : Union[List[float], List[List[float]]]
        Initial values for state variables [X, S, P, I].
    true_model : callable, optional
        Model function to generate data, by default true_model_day2.
    n_timepoints : int, optional
        Number of time points to generate, by default 20.
    noise_level : float, optional
        Standard deviation of noise relative to signal.

    Returns
    -------
    List[Dict[str, Any]]
        Training data as list of dictionaries containing time points and noisy data

    Raises
    ------
    TypeError
        If initial_conditions is not a list.
    """
    if not isinstance(initial_conditions, list):
        raise TypeError("initial_conditions must be a list")

    if isinstance(initial_conditions[0], (int, float)):
        initial_conditions = [initial_conditions]

    t = np.linspace(0, 10, n_timepoints)
    training_data = []

    for i, y0 in enumerate(initial_conditions):
        # Generate clean data
        solution = odeint(true_model, y0, t)

        # Add noise
        noise = np.random.normal(0, noise_level, solution.shape)
        noisy_data = solution + noise * solution

        # Store data
        training_data.append({"t": t, "data": noisy_data, "initial_conditions": y0})

    return training_data


def generate_test_data(
    test_conditions: Union[List[float], List[List[float]]],
    true_model: callable,
) -> List[Dict[str, Any]]:
    """
    Generate test data with specific conditions.

    Parameters
    ----------
    true_model : callable, optional
        Model function to generate data, by default true_model_day2.
    test_conditions : Union[List[float], List[List[float]]]
        Initial values for state variables [X, S, P, I].

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing time points and test data.
    """
    t = np.linspace(0, 15, 30)  # Longer time period
    test_data = []

    for y0 in test_conditions:
        solution = odeint(true_model, y0, t)
        test_data.append({"t": t, "data": solution, "initial_conditions": y0})

    return test_data


def fitness_function(
    mask: List[int],
    params: List[float],
    training_data: List[Dict[str, Any]],
) -> Tuple[float, Optional[List[float]]]:
    """
    Calculate fitness of a solution.

    Parameters
    ----------
    mask : List[int]
        Binary mask indicating which growth terms to include.
    params : List[float]
        Model parameters [mu_max, Ks, Ki, Yxs, Kp].
    training_data : List[Dict[str, Any]]
        Training data to evaluate fitness against.

    Returns
    -------
    Tuple[float, Optional[List[float]]]
        Tuple containing:
        - Negative total error (including complexity penalty)
        - List of errors for each experiment (or None if error occurred)
    """
    total_error = 0
    n_active_terms = sum(mask)

    errors_by_experiment = []
    for experiment in training_data:
        t = experiment["t"]
        y_true = experiment["data"]
        y0 = experiment["initial_conditions"]

        try:
            y_pred = odeint(
                lambda y, t: candidate_models(y, t, params, mask),
                y0,
                t,
            )
            error = np.mean((y_true - y_pred) ** 2)
            total_error += error
            errors_by_experiment.append(error)
        except:
            return float("-inf"), None

    return -total_error, errors_by_experiment


def plot_results(
    training_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    best_individual: Dict[str, Any],
    test_predictions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Plot results comparing training data, test data, and predictions with custom styling.

    Parameters
    ----------
    training_data : List[Dict[str, Any]]
        List of dictionaries containing training data with keys:
        't', 'data', and 'initial_conditions'.
    test_data : List[Dict[str, Any]]
        List of dictionaries containing test data with keys:
        't', 'data', and 'initial_conditions'.
    best_individual : Dict[str, Any]
        Dictionary containing the best solution with keys:
        'mask' and 'params'.
    test_predictions : Optional[List[Dict[str, Any]]], optional
        List of dictionaries containing model predictions for test data,
        by default None.
    """
    plt.figure(figsize=(15, 10))
    plt.suptitle("Model Performance on Training and Test Data", fontsize=16)

    variables = ["Biomass", "Substrate", "Product", "Inhibitor"]
    colors_training = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#e377c2"]
    colors_test = ["#9467bd", "#8c564b", "#7f7f7f", "#bcbd22", "#17becf"]

    for idx, var in enumerate(variables):
        plt.subplot(2, 2, idx + 1)

        # Plot training data and predictions
        for i, exp in enumerate(training_data):
            color = colors_training[i]
            # Training data (circles)
            plt.scatter(
                exp["t"],
                exp["data"][:, idx],
                color=color,
                marker="o",
                alpha=0.6,
                label=f"Training Set {i + 1}",
            )

            # Training predictions (dashed line)
            t = exp["t"]
            y0 = exp["initial_conditions"]
            y_pred = odeint(
                lambda y, t: candidate_models(
                    y, t, best_individual["params"], best_individual["mask"]
                ),
                y0,
                t,
            )
            plt.plot(
                t,
                y_pred[:, idx],
                color=color,
                linestyle="--",
                alpha=0.8,
                label=f"Training Prediction {i + 1}",
            )

        # Plot test data and predictions
        if test_predictions:
            for i, (exp, pred) in enumerate(zip(test_data, test_predictions)):
                color = colors_test[i]
                # Test data (squares)
                plt.scatter(
                    exp["t"],
                    exp["data"][:, idx],
                    color=color,
                    marker="s",
                    alpha=0.6,
                    label=f"Test Set {i + 1}",
                )
                # Test predictions (dashed line)
                plt.plot(
                    pred["t"],
                    pred["prediction"][:, idx],
                    color=color,
                    linestyle="--",
                    alpha=0.8,
                    label=f"Test Prediction {i + 1}",
                )

        plt.xlabel("Time")
        plt.ylabel(var)
        if idx == 0:  # Only show legend for first subplot
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
    return


def _plot_evolution(
    fitness_history: List[float],
    param_history: List[List[float]],
) -> None:
    """
    Plot the evolution of fitness and parameters.

    Parameters
    ----------
    fitness_history : List[float]
        History of best fitness values.
    param_history : List[List[float]]
        History of parameter values.
    """
    # Plot fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Fitness Evolution")
    plt.grid(True)
    plt.show()

    # Plot parameter evolution
    param_history = np.array(param_history)
    plt.figure(figsize=(12, 8))
    param_names = ["mu_max", "Ks", "Ki", "Yxs", "Kp"]
    for i in range(len(param_names)):
        plt.plot(param_history[:, i], label=param_names[i])
    plt.xlabel("Generation")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Evolution")
    plt.legend()
    plt.grid(True)
    plt.show()
    return
