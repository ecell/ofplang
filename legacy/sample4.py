import numpy as np
import matplotlib.pyplot as plt
from ofplang.definitions import Definitions
from ofplang.protocol import Protocol
from ofplang.runner import Runner
from ofplang.executors import Simulator, GaussianProcessExecutor

# Load definitions and protocol
definitions = Definitions('./manipulate.yaml')
protocol = Protocol("./sample1.yaml")
runner = Runner(protocol, definitions, executor=Simulator())

# Define training inputs
inputs_training = {"volume": {"value": np.linspace(0, 200, 96), "type": "Array[Float]"}}
experiment = runner.run(inputs=inputs_training)

# Train Gaussian Process Executor
executor = GaussianProcessExecutor()
executor.teach(runner.model, experiment)
print(f"Mean: {executor.mean}")
print(f"Covariance: {executor.covariance}")

# Define new inputs
inputs = {"volume": {"value": np.linspace(0, 200, 96), "type": "Array[Float]"}}
new_experiment = runner.run(inputs=inputs)
prediction = runner.run(inputs=inputs, executor=executor)

# Extract mean and covariance
mean_gpytorch = executor.mean
covariance_gpytorch = executor.covariance

# Compute standard deviation from covariance
std_gpytorch = np.sqrt(np.diagonal(covariance_gpytorch, axis1=0, axis2=1))

# Extract training data for plotting
X_train = inputs_training["volume"]["value"]
y_train = np.array(experiment.output["data"]["value"])

# Generate X and y for true function (for example)
X = np.linspace(0, 200, 100)
y = X * np.sin(X)

for idx in range(3):
    x = new_experiment.output["data"]["value"][idx]
    y = prediction.output["data"]["value"][idx]
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    vmin, vmax = vmin - (vmin + vmax) * 0.05, vmax + (vmin + vmax) * 0.05
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot((vmin, vmax), (vmin, vmax), "k--")
    ax.plot(x, y, 'k.')
    #ax.plot(inputs["volume"]["value"], x, 'k.')
    #ax.plot(inputs["volume"]["value"], y, 'r.')
    ax.set_xlabel(r'$Experiment$')
    ax.set_ylabel(r'$Prediction$')
    # ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.savefig(f'comparison{idx}.png')

for idx in range(mean_gpytorch.shape[1]):
    plt.figure(figsize=(7, 6))
    
    # Plot observations
    plt.scatter(X_train, y_train[idx], label="Observations", color='red')
    
    # Plot mean prediction
    plt.plot(inputs["volume"]["value"], prediction.output["data"]["value"][idx], label="Mean prediction", color='blue')
    
    # Plot 95% confidence interval
    lower_gpytorch = prediction.output["data"]["value"][idx] - 1.96 * std_gpytorch[idx]
    upper_gpytorch = prediction.output["data"]["value"][idx] + 1.96 * std_gpytorch[idx]
    plt.fill_between(X_train, lower_gpytorch, upper_gpytorch, alpha=0.5, label="95% confidence interval", color='blue')
    
    plt.legend()
    plt.xlabel("Volume")
    plt.ylabel("Output Value")
    plt.title(f"Gpytorch Gaussian Process Regression with Confidence Interval (Output {idx})")
    plt.tight_layout()
    plt.savefig(f'comparison{idx + 3}.png')  # Save as Comparison 3, 4, 5