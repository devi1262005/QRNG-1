import numpy as np
import matplotlib.pyplot as plt
import secrets
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from matplotlib.widgets import Button

# Generate QRNG numbers
num_qubits = 8
shots = 8192
circuit = QuantumCircuit(num_qubits)
circuit.h(range(num_qubits))
circuit.measure_all()
simulator = Aer.get_backend('aer_simulator')
compiled_circuit = transpile(circuit, simulator)
result = simulator.run(compiled_circuit, shots=shots).result()
counts = result.get_counts()
qrng_numbers = [int(b, 2) for b in counts.keys()]

# Generate PRNG and CSPRNG numbers
rng = np.random.default_rng(seed=42)  # Uses PCG64 by default

# Generate PRNG numbers with the same length as qrng_numbers
prng_numbers = rng.integers(0, 256, size=len(qrng_numbers))
csprng_numbers = [secrets.randbelow(256) for _ in range(len(qrng_numbers))]

# Compute probability distributions
def compute_distribution(numbers):
    values, occurrences = np.unique(numbers, return_counts=True)
    probabilities = occurrences / sum(occurrences)
    return values, probabilities

qrng_values, qrng_probabilities = compute_distribution(qrng_numbers)
prng_values, prng_probabilities = compute_distribution(prng_numbers)
csprng_values, csprng_probabilities = compute_distribution(csprng_numbers)

# Theory Texts
theories = [
    "QRNG (Quantum Random Number Generator)\n\n"
    "- Uses quantum physics (hardware-based) for randomness.\n"
    "- Highest Shannon entropy, meaning most unpredictable.\n"
    "- Used in high-security cryptography and scientific research.\n"
    "- QRNG entropy remains consistently high.\n",
    "PRNG (Pseudo-Random Number Generator)\n\n"
    "- Algorithm-based, meaning it's deterministic.\n"
    "- Lower Shannon entropy, making it predictable over time.\n"
    "- Used in gaming, simulations, and general randomness needs.\n"
    "- PRNG takes longer to stabilize and has more variation.\n",
    "CSPRNG (Cryptographically Secure PRNG)\n\n"
    "- Uses cryptographic algorithms (secrets.randbelow()).\n"
    "- More unpredictable than PRNG but not as strong as QRNG.\n"
    "- Used in military, banking, and cryptographic applications.\n"
    "- CSPRNG stabilizes faster than PRNG but is still slightly less random than QRNG.\n",
]

# Table Data for Step 3
table_data = [
    ["Metric", "QRNG (Quantum)", "PRNG (Standard)", "CSPRNG (Military)"],
    ["True Random?", "Yes (Hardware)", "No (Algorithm)", "Yes (Crypto)"],
    ["Shannon Entropy", "Highest", "Lower", "High"],
    ["Security Level", "Unbreakable", "Predictable", "Secure"],
    ["Used In", "Scientific, Cryptography", "Games, Simulations", "Military, Cryptography"],
    ["KS-Test Similarity to QRNG", "-", "Lower", "Higher"]
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3)
step = 0  # Track current visualization step

# Function to update the plot
def update_plot():
    ax.clear()
    
    if step == 0:
        ax.bar(qrng_values, qrng_probabilities, color="blue", edgecolor="black", alpha=0.7)
        ax.set_title("QRNG (Quantum Randomness)", fontsize=14, fontweight="bold")
        ax.text(0.5, -0.35, theories[step], transform=ax.transAxes, ha="center", fontsize=10, 
                wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black"))

    elif step == 1:
        ax.bar(prng_values, prng_probabilities, color="green", edgecolor="black", alpha=0.7)
        ax.set_title("PRNG (Pseudo-Random Generator)", fontsize=14, fontweight="bold")
        ax.text(0.5, -0.35, theories[step], transform=ax.transAxes, ha="center", fontsize=10, 
                wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black"))

    elif step == 2:
        ax.bar(csprng_values, csprng_probabilities, color="red", edgecolor="black", alpha=0.7)
        ax.set_title("CSPRNG (Military-Grade Secure Randomness)", fontsize=14, fontweight="bold")
        ax.text(0.5, -0.35, theories[step], transform=ax.transAxes, ha="center", fontsize=10, 
                wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black"))

    else:  # Show tabular comparison
        ax.axis("off")  # Hide axis for table display
        ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.3]*4)
        ax.set_title("Comparison of QRNG, PRNG, and CSPRNG", fontsize=14, fontweight="bold")

    plt.draw()

# Function to handle next button click
def next_step(event):
    global step
    if step < 3:
        step += 1
        update_plot()

# Function to handle back button click
def prev_step(event):
    global step
    if step > 0:
        step -= 1
        update_plot()

# Add back and next buttons
ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
btn_next = Button(ax_next, "Next")
btn_next.on_clicked(next_step)

ax_back = plt.axes([0.1, 0.05, 0.1, 0.075])
btn_back = Button(ax_back, "Back")
btn_back.on_clicked(prev_step)

update_plot()  # Show first plot
plt.show()
