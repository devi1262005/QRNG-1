import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, entropy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# 🎯 Parameters
num_qubits = 8  # 8-bit random numbers (0-255)
shots = 8192  # High sample size for accuracy

# 🎲 Quantum Circuit for QRNG
circuit = QuantumCircuit(num_qubits)
circuit.h(range(num_qubits))  # Apply Hadamard gates
circuit.measure_all()

simulator = Aer.get_backend('aer_simulator')
compiled_circuit = transpile(circuit, simulator)
result = simulator.run(compiled_circuit, shots=shots).result()
counts = result.get_counts()

# Convert QRNG results to numerical values
binary_samples = list(counts.keys())
qrng_numbers = [int(b, 2) for b in binary_samples]

# 🔢 Classical Random Numbers (CRNG) for Comparison
crng_numbers = np.random.randint(0, 256, len(qrng_numbers))

# 📈 Compute Probability Distributions
qrng_values, qrng_occurrences = np.unique(qrng_numbers, return_counts=True)
qrng_probabilities = qrng_occurrences / sum(qrng_occurrences)

crng_values, crng_occurrences = np.unique(crng_numbers, return_counts=True)
crng_probabilities = crng_occurrences / sum(crng_occurrences)

# 📊 Compute Shannon Entropy for QRNG & CRNG
qrng_entropy = -np.sum(qrng_probabilities * np.log2(qrng_probabilities))
crng_entropy = -np.sum(crng_probabilities * np.log2(crng_probabilities))

# 🔍 KS Test (QRNG vs CRNG)
ks_stat, ks_p_value = ks_2samp(qrng_numbers, crng_numbers)

# 🔵 PLOTTING DISTRIBUTIONS 🔵
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 📊 Histogram: QRNG Distribution
axs[0].bar(qrng_values, qrng_probabilities, color="skyblue", alpha=0.7, edgecolor="black", label="QRNG")
axs[0].bar(crng_values, crng_probabilities, color="red", alpha=0.3, edgecolor="black", label="CRNG")
axs[0].set_xlabel("Random Number (0-255)")
axs[0].set_ylabel("Probability")
axs[0].set_title("QRNG vs CRNG Distribution")
axs[0].legend()

# 📈 Entropy Progression Over Time
sample_sizes = np.linspace(100, shots, 20, dtype=int)
qrng_entropy_progression = [entropy(np.unique(qrng_numbers[:s], return_counts=True)[1] / s) for s in sample_sizes]
crng_entropy_progression = [entropy(np.unique(crng_numbers[:s], return_counts=True)[1] / s) for s in sample_sizes]

axs[1].plot(sample_sizes, qrng_entropy_progression, label="QRNG Entropy", color="blue", linewidth=2)
axs[1].plot(sample_sizes, crng_entropy_progression, label="CRNG Entropy", color="red", linestyle="dashed", linewidth=2)
axs[1].set_xlabel("Number of Samples")
axs[1].set_ylabel("Shannon Entropy (bits)")
axs[1].set_title("Entropy Stabilization: QRNG vs CRNG")
axs[1].legend()

plt.suptitle(f"Quantum Randomness vs Classical Randomness\nKS Test p-value = {ks_p_value:.4f} (Higher is better)")
plt.show()

# 🎯 Print Key Findings
print(f"🔵 QRNG Shannon Entropy: {qrng_entropy:.4f} bits")
print(f"🔴 CRNG Shannon Entropy: {crng_entropy:.4f} bits")
print(f"🔍 Kolmogorov-Smirnov Test p-value: {ks_p_value:.4f} (Closer to 1 means QRNG matches true randomness)")
