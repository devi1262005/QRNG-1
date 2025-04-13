from flask import Flask, jsonify
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

app = Flask(__name__)

def generate_qrng_number():
    """Generates a quantum random number in the range 1-13"""
    num_qubits = 4  # 4 qubits can give values from 0 to 15
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))  # Apply Hadamard gate for superposition
    circuit.measure_all()

    simulator = Aer.get_backend('aer_simulator')
    compiled_circuit = transpile(circuit, simulator)
    result = simulator.run(compiled_circuit, shots=1).result()

    counts = result.get_counts()
    qrng_number = int(list(counts.keys())[0], 2)  # Convert binary to integer

    # Map the number to 1-13 (ignore 14, 15)
    while qrng_number > 13 or qrng_number == 0:
        qrng_number = generate_qrng_number()  # Rerun if out of range

    return qrng_number

@app.route('/qrng_nos', methods=['POST'])
def get_qrng_no():
    """Returns a single QRNG number in the range 1-13"""
    qrng_number = generate_qrng_number()
    return jsonify({"qrng_number": qrng_number})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
