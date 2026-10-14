import numpy as np
from typing import Callable
from pathlib import Path

Arr = np.ndarray
grad_tracking_enabled = True
data_path = Path("data/MNIST/raw")


###################################################
# Tensor class with Autograd engine
###################################################
"""
To compute backprop, every 'node' in the computational graph takes a grad_out (the gradient of everything
to the right of it), and uses that to compute the grads of its parents, the nodes used to create it.
E.g. [a] [b] -> [c = a + b] -> out
grad out/c = 1 (because a change to [c] affects out 1:1).
so [c] takes out_grad = 1, and since [a] and [b] affect [c] 1:1, c._backward returns (1, 1) which can be sent
to [a] and [b] as their grad_out.
"""
class Tensor:
    '''
    Tensor: (Replicates parts of the functionality of torch.Tensor)
    '''
    array: Arr
    "The underlying array. Can be shared across multiple Tensors"
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop"
    grad: "Tensor | None"
    "Backpropagation will accumulate gradients into this field."
    parents: tuple["Tensor",...] 
    "Ordered list of parents. Parents are the nodes that are used to compute this Tensor"
    "e.g. if c = a + b, a and b are parents of c"
    _backward: Callable | None
    "Back func that can be used to compute the gradient of the Tensors that where inputs to this one"
    "Takes out_grad, the grad of everything until now, and returns the grad of each parent"

    def __init__(self, array: Arr | list, requires_grad=False):
        # Array object - convert to np.array if needed
        self.array = array if isinstance(array, Arr) else np.array(array)
        # Convert array to float32 if needed
        if self.array.dtype == np.float64 or np.issubdtype(self.array.dtype, np.integer):
            self.array = self.array.astype(np.float32)

        self.requires_grad = requires_grad
        self.grad = None
        self.parents = ()
        self._backward = None
        # These fields get filled during the forward pass as instructions for backprop

    '''
    Custom Tensor operations.
    Anything that can uses primitives, e.g. subtract = self + -other
    '''
    def __neg__(self) -> "Tensor":
        # Forward function
        out_array = -self.array

        # Backwards function
        def _backward(grad_out):
            return (-grad_out,)

        # Make output: create out Tensor, if grad tracking needed, add parents and backward func
        return _make_output(out_array, (self,), _backward)

    def __mul__(self, other) -> "Tensor":
        # Process other into tensor
        other = to_tensor(other)

        # Forward function
        out_array = self.array * other.array

        def _backward(grad_out):
            return (
                unbroadcast(grad_out * other.array, self.array),
                unbroadcast(grad_out * self.array, other.array)
            )

        # Make output
        return _make_output(out_array, (self, other), _backward)

    def __add__(self, other) -> "Tensor":
        # Process other into tensor
        other = to_tensor(other)

        # Forward function
        out_array = self.array + other.array

        def _backward(grad_out):
            return (
                unbroadcast(grad_out, self.array),
                unbroadcast(grad_out, other.array),
            )

        return _make_output(out_array, (self, other), _backward)

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __sub__(self, other) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other) -> "Tensor":
        return other + (-self)

    def __pow__(self, n) -> "Tensor":
        assert isinstance(n, (int, float)), "only number exponents supported"

        # Forward function
        out_array = self.array ** n

        # Backwards function
        def _backward(grad_out):
            return (grad_out * n * self.array ** (n - 1),) 

        return _make_output(out_array, (self,), _backward)

    def __truediv__(self, other) -> "Tensor":
        return self * other ** -1

    def __rtruediv__(self, other) -> "Tensor":
        return other * self ** -1

    def sum(self, axis=None) -> "Tensor":
        # Forward function
        out_array = self.array.sum(axis=axis)
    
        def _backward(grad_out):
            if axis is not None:
                grad_out = np.expand_dims(grad_out, axis)
            return (np.broadcast_to(grad_out, self.array.shape).copy(),)

        return _make_output(out_array, (self,), _backward)

    def log(self) -> "Tensor":
        # Forward function
        out_array = np.log(self.array)

        # Backwards function
        def _backward(grad_out):
            return (grad_out / self.array,)

        return _make_output(out_array, (self,), _backward)

    def exp(self) -> "Tensor":
        # Forward function
        out_array = np.exp(self.array)

        # Backwards function
        def _backward(grad_out):
            return (grad_out * out_array,)

        return _make_output(out_array, (self,), _backward)

    def relu(self) -> "Tensor":
        # Forward function
        out_array = np.maximum(self.array, 0)

        # Backwards function
        def _backward(grad_out):
            return (grad_out * (self.array > 0),) # 1 when > 0 , 0 when <= 0

        return _make_output(out_array, (self,), _backward)

    def __matmul__(self, other) -> "Tensor":
        """Only 2d Tensors"""
        assert isinstance(other, Tensor), "matmul needs a Tensor"
        # Forward function
        out_array = self.array @ other.array

        def _backward(grad_out):
            return (
                grad_out @ other.array.T,
                self.array.T @ grad_out
            )

        return _make_output(out_array, (self, other), _backward)

    @property
    def T(self) -> "Tensor":
        # Forward
        out_array = self.array.T

        def _backward(grad_out):
            return (grad_out.T,)

        return _make_output(out_array, (self,), _backward)

    def item(self) -> float:
        return self.array.item()

    
    '''Backward'''
    def backward(self, end_grad: Arr | None = None):
        """Accumulates gradients in the grad field of each leaf node.
        self:
            The rightmost node in the computation graph. If it contains more than one element, end_grad
            must be provided.
        end_grad:
            An array of the same shape as the end node (self). Set to 1 if not specified and has only one
            element.
        """
        # Get value of the end grad (how much the last node effects the output)
        end_grad_arr = np.ones_like(self.array) if end_grad is None else end_grad

        # Create dict to store gradients
        grads: dict[Tensor, Arr] = {self: end_grad_arr}

        # Create and cycle computational graph backwads from this Tensor node
        for node in reversed(topological_sort(self)):
            # Retrieve and pop the stored grad of everything to the right of this branch
            node_grad = grads.pop(node)

            # If this is a leaf node and requires grad, set grad
            if not node.parents:
                if node.requires_grad:
                    # if this is the first grad, just store
                    if node.grad is None:
                        node.grad = Tensor(node_grad)
                    # If there is already a grad stored, sum
                    else:
                        node.grad.array += node_grad

            # If node not leaf
            else:
                # Compute parent grads
                parent_grads = node._backward(node_grad)
                for parent, parent_grad in zip(node.parents, parent_grads):
                    # Save new grad + old grad | 0 if none
                    grads[parent] = grads.get(parent, 0) + parent_grad

    def __repr__(self):
        return f"Tensor({self.array}, requires_grad={self.requires_grad})"

# Helpers
def _make_output(array: Arr, parents: tuple, backward: Callable) -> "Tensor":
    requires_grad = grad_tracking_enabled and any(p.requires_grad for p in parents)
    out = Tensor(array, requires_grad)
    if requires_grad:
        out.parents = parents
        out._backward = backward
    return out

def to_tensor(x: Arr | Tensor | float) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)

def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by
        broadcasting rules.
    """
    # First sum over prepended dims
    n_dims_to_sum = len(broadcasted.shape) - len(original.shape)
    broadcasted = broadcasted.sum(axis=tuple(range(n_dims_to_sum)))

    # Sum over dims which were 1
    dims_to_sum = tuple([i for i, (o, b) in enumerate(zip(original.shape, broadcasted.shape)) if o == 1 and b > 1])
    broadcasted = broadcasted.sum(axis=dims_to_sum, keepdims=True)

    assert broadcasted.shape == original.shape
    return broadcasted

def topological_sort(node: Tensor) -> list[Tensor]:
    """
    Returns a list of node's ancestors in topological order from past
    to present.
    E.g. if [x] = [a] + [b] and [y] = [x] + [c], sort(y) will return [a, b, x, c, y]
    since y uses x and x uses b etc
    """
    result: list[Tensor] = [] # List of all dependancies from start to end
    visited: set[Tensor] = set() # Same contents as result but as a more searchable set

    def visit(current):
        if current in visited:
            return

        for parent in current.parents:
            visit(parent)

        visited.add(current)
        result.append(current)

    visit(node)
    return result


############################################################
# Neural Network init
############################################################
class Parameter(Tensor):
    """Wrapper for Tensor that declares it a trainable parameter"""
    def __init__(self, tensor: Tensor, requires_grad=True):
        """Share the array with the provided tensor."""
        return super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"

    
class Module:
    """
    Stores submodules and trainable parameters.
    Designed to simplify param lookup throughout a network
    """
    _modules: dict[str, "Module"]
    "Submodules of a module group, e.g. MLP module has linear1, linear2..."
    _parameters: dict[str, Parameter]
    "All parameters in Module"

    def __init__(self):
        self._modules: dict[str, "Module"] = {}
        self._parameters: dict[str, Parameter] = {}

    def modules(self) -> list["Module"]:
        """Return the direct child modules of this module, not including self."""
        return list(self._modules.values())

    def parameters(self, recurse: bool = True) -> list:
        """Returns all params in Module, and submodules if recurse"""
        params = list(self._parameters.values())
        if recurse:
            for module in self.modules():
                params.extend(module.parameters(recurse=True))
        return params

    def __setattr__(self, name, value):
        """
        If value is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        """
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self.__dict__["_parameters"]:
            return self._parameters[name]
        elif name in self.__dict__["_modules"]:
            return self._modules[name]
        else:
            raise AttributeError(f"{name} not found in Module")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward!")

    def extra_repr(self):
        return ""
    
    def __repr__(self):
        _indent = lambda s_, nSpaces: s_.replace("\n", "\n" + " " * nSpaces)
        lines = [f"({key}): {_indent(repr(module), 2)}" for key, module in self._modules.items()]
        return "".join(
            [
                self.__class__.__name__ + "(" + self.extra_repr(),
                "\n  " + "\n  ".join(lines) + "\n" if lines else "",
                ")",
            ]
        )


class Linear(Module):
    """
    Simple linear transformation. Stores weights and optional biases
    The fields are named `weight` and `bias` for compatibility with PyTorch.
    """
    weight: Parameter
    bias: Parameter | None

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Kaiming bounds on parameter randomness
        kaiming_bound = 1/np.sqrt(in_features)
        # Init random weights, shift them to be between the bounds
        weights_array = np.random.uniform(-kaiming_bound, kaiming_bound, (self.out_features, self.in_features))
        self.weight = Parameter(Tensor(weights_array))

        # Bias
        if bias:
            bias_array = np.random.uniform(-kaiming_bound, kaiming_bound, out_features)
            self.bias = Parameter(Tensor(bias_array))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, in_features)
        Return: shape (batch, out_features)
        """
        # Matmul the imputs to the weights 
        activations = x @ self.weight.T

        # Add bias if there is one
        if self.bias is not None:
            activations += self.bias

        return activations

    def extra_repr(self) -> str:
        return f"In features: {self.in_features}, Out features: {self.out_features}, Bias: {self.bias is not None}"

#########################################################
# MLP assembly
#########################################################
class MLP(Module):
    """
    """
    def __init__(self):
        super().__init__()
        # Module structure
        self.linear1 = Linear(784, 128)

        self.linear2 = Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear1(x)
        out = out.relu()
        out = self.linear2(out)
        return out

##########################################################
# Training parts
##########################################################
def cross_entropy(logits: Tensor, true_labels: Arr) -> Tensor:
    """Like torch.nn.functional.cross_entropy'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    """
    batch_size = logits.array.shape[0]

    # Convert labels to onehot (2 -> [0, 0, 1, 0...])
    onehot_labels = Tensor(np.eye(logits.array.shape[1])[true_labels])

    # Stabilize the logits by subtracting the max
    row_maxes = Tensor(logits.array.max(axis=1, keepdims=True), requires_grad=False)
    stabilized_logits = logits - row_maxes

    #
    soft_max_score = stabilized_logits.exp().sum(axis=1).log()      # (batch,)
    correct_score  = (stabilized_logits * onehot_labels).sum(axis=1)  # (batch,)
    loss = (soft_max_score - correct_score).sum() / batch_size        # one number

    return loss

class NoGrad:
    """Context manager that disables grad inside a block. Like torch.no_grad."""
    was_enabled: bool

    def __enter__(self):
        """
        Method which is called whenever the context manager is entered, i.e. at the start of the
        `with NoGrad():` block. This disables gradient tracking (but stores the value it had before,
        so we can set it back to this on exit).
        """
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        """
        Method which is called whenever we exit the context manager. This sets the global
        `grad_tracking_enabled` variable back to the value it had before we entered the context
        manager.
        """
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled
   
class SGD:
    def __init__(self, params: list[Parameter], lr: float):
        """Vanilla SGD with no additional features."""
        self.params = params
        self.lr = lr

    def zero_grad(self) -> None:
        """Iterates through params, and sets all grads to None."""
        for param in self.params:
            param.grad = None

    def step(self) -> None:
        """Iterates through params, and updates each of them by subtracting `param.grad * lr`."""
        with NoGrad():
            for param in self.params:
                param.array -= param.grad.array * self.lr


##########################################################
# Training
##########################################################
def load_mnist_images(path: str) -> Arr:
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)  # skip 16-byte header
    return data.reshape(-1, 784).astype(np.float32) / 255 # flatten + scale to 0–1

def load_mnist_labels(path: str) -> Arr:
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)  # 8-byte header


def train_epoch(
    model: MLP,
    images: Arr,
    labels: Arr,
    optimizer: SGD,
    batch_size: int,
):
    loss_list = []

    # Shuffle images in sync with labels
    perm = np.random.permutation(len(images))
    images, labels = images[perm], labels[perm]

    # Split into batches, For batch:
    for start in range(0, len(images), batch_size):
        batch_images = Tensor(images[start : start + batch_size], requires_grad=False)
        batch_labels = labels[start : start + batch_size]

        # Forward
        optimizer.zero_grad()
        logits = model(batch_images)

        # Backward
        loss = cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step()

        # Save loss
        loss_list.append(loss.item())

    return loss_list

def evaluate(model: MLP, images: Arr, labels: Arr) -> tuple[float, float]:
    """Returns (average loss, accuracy) over the whole dataset."""
    with NoGrad():
        logits = model(Tensor(images))
        loss = cross_entropy(logits, labels).item()

        predictions = logits.array.argmax(axis=1)    # predicted digit per image
        accuracy = (predictions == labels).mean()    # fraction correct

    return loss, float(accuracy)
    
def train_loop(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.1,
    on_epoch_end: Callable | None = None
):
    # Get data
    print("Loading data")
    train_images = load_mnist_images(data_path / "train-images-idx3-ubyte" )
    train_labels = load_mnist_labels(data_path / "train-labels-idx1-ubyte")
    test_images  = load_mnist_images(data_path / "t10k-images-idx3-ubyte")
    test_labels  = load_mnist_labels(data_path / "t10k-labels-idx1-ubyte")

    model = MLP()
    optimizer = SGD(model.parameters(), lr=lr)
    train_loss_history = []
    test_acc_history = []

    # Train loop
    print("Starting train loop")
    for epoch_num in range(epochs):
        train_losses = train_epoch(model, train_images, train_labels, optimizer, batch_size)
        test_loss, test_acc = evaluate(model, test_images, test_labels)
        # Send epoch data to given on_epoch_end function and to history
        train_loss_history.append(train_losses)
        test_acc_history.append(test_acc)
        if on_epoch_end is not None:
            on_epoch_end(epoch_num, train_losses, test_loss, test_acc)

    return model, train_loss_history, test_acc_history


if __name__ == "__main__":
    def print_epoch_callback(epoch_num, train_losses, test_loss, test_acc):
        print(
            f"Epoch {epoch_num + 1}: "
            f"train loss {np.mean(train_losses):.3f}, "
            f"test loss {test_loss:.3f}, "
            f"test acc {test_acc:.1%}"
        )
    # Train
    model, train_loss_history, test_acc_history = train_loop(epochs=2, on_epoch_end=print_epoch_callback)
    print("Training complete!")

    # Plot results
    import matplotlib.pyplot as plt

    batch_losses = np.concatenate(train_loss_history)
    smoothed = np.convolve(batch_losses, np.ones(50) / 50, mode="valid")
    epoch_nums = range(1, len(test_acc_history) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(batch_losses, alpha=0.3, label="per batch")
    ax_loss.plot(range(49, len(batch_losses)), smoothed, label="50-batch average")
    ax_loss.set(title="Training loss", xlabel="Batch", ylabel="Loss")
    ax_loss.legend()

    ax_acc.plot(epoch_nums, np.array(test_acc_history) * 100, marker="o")
    ax_acc.set(title="Test accuracy", xlabel="Epoch", ylabel="Accuracy (%)", xticks=list(epoch_nums))

    plt.tight_layout()
    plt.show()