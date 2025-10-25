"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import time


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x).sigmoid()
        return x


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        batch = x.shape[0]
        in_size = x.shape[1]
        output = self.bias.value.zeros((batch, self.out_size)) + self.bias.value
        for i in range(in_size):
            selector = []
            for k in range(in_size):
                selector.append(1.0 if k == i else 0.0)
            selector_tensor = minitorch.tensor([selector], backend=x.backend)
            x_col = (x * selector_tensor).sum(1).view(batch, 1)
            selector_col = selector_tensor.view(in_size, 1)
            w_row = (self.weights.value * selector_col).sum(0).view(1, self.out_size)
            output = output + (x_col * w_row)
        return output

def default_log_fn(epoch, total_loss, correct, losses, epoch_time):
    print(f"Epoch {epoch} loss {total_loss} correct {correct} time {epoch_time:.4f}s")


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)
        losses = []
        epoch_times = []
        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()
            total_loss = 0.0
            correct = 0
            optim.zero_grad()
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)
            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)
            optim.step()
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                avg_time = sum(epoch_times[-10:]) / min(10, len(epoch_times))
                log_fn(epoch, total_loss, correct, losses, avg_time)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 100
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)