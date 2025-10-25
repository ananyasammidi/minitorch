from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Uses the central difference formula: f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant for finite difference approximation

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    values_positive = list(vals)
    values_negative = list(vals)

    values_positive[arg] = vals[arg] + epsilon
    values_negative[arg] = vals[arg] - epsilon
    return (f(*values_positive) - f(*values_negative)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Add the derivative to the total gradient using this function"""
        pass

    @property
    def unique_id(self) -> int:
        """Defining unique_id identifier for each variable"""
        pass

    def is_leaf(self) -> bool:
        """Checking if the variable is a leaf"""
        pass

    def is_constant(self) -> bool:
        """Verifiying if the variable is a constant"""
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """Collecting the parents of the variable"""
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Defining the chain rule"""
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.

    Hints:
        - Use depth-first search (DFS) to visit nodes
        - Track visited nodes to avoid cycles (use node.unique_id)
        - Return nodes in reverse order (dependencies first)

    """
    # TODO: Implement for Task 1.4.
    order = []
    visited = set()

    def dfs(node: Variable) -> None:
        if node.unique_id in visited or node.is_constant():
            return
        if not node.is_leaf():
            for m in node.parents:
                if not m.is_constant():
                    dfs(m)
        visited.add(node.unique_id)
        order.insert(0, node)
    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    Hints:
        - First get all nodes in topological order using topological_sort()
        - Create a dictionary to store derivatives for each node (keyed by unique_id)
        - Initialize the starting node's derivative to the input deriv
        - Process nodes in the topological order (which is already correct for backprop)
        - For leaf nodes: call node.accumulate_derivative(derivative)
        - For non-leaf nodes: call node.chain_rule(derivative) to get parent derivatives
        - Sum derivatives when the same parent appears multiple times

    """
    # TODO: Implement for Task 1.4.
    nodes = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for node in nodes:
        node_derivative = derivatives.get(node.unique_id, 0.0)
        if node.is_leaf():
            node.accumulate_derivative(node_derivative)
        else:
            for parent, parent_derivative in node.chain_rule(node_derivative):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = parent_derivative
                else:
                    derivatives[parent.unique_id] += parent_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returning a tuple of saved values"""
        return self.saved_values
