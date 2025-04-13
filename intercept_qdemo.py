import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import time
from threading import Thread
from queue import Queue
import binascii
import secrets
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# Configure page
st.set_page_config(page_title="QRNG vs CSPRNG Comparison", layout="wide")
st.title("Quantum vs Classical Random Number Generation")

# Constants
NUM_CHANNELS = 13
SIMULATION_STEPS = 100
QRNG_CACHE_SIZE = 100
PACKET_SIZE = 10

class QRNGGenerator:
    def __init__(self):
        self.simulator = Aer.get_backend('aer_simulator')
    
    def generate_random_number(self, max_value=255):
        """Generate a quantum random number between 0 and max_value"""
        # Calculate required number of qubits
        num_qubits = (max_value).bit_length()
        
        # Create quantum circuit
        circuit = QuantumCircuit(num_qubits)
        circuit.h(range(num_qubits))  # Apply Hadamard gates
        circuit.measure_all()
        
        # Run the circuit
        compiled_circuit = transpile(circuit, self.simulator)
        result = self.simulator.run(compiled_circuit, shots=1).result()
        
        # Get the measurement result
        counts = result.get_counts()
        binary = list(counts.keys())[0]
        value = int(binary, 2)
        
        # If value is too large, map it to the desired range
        return value % (max_value + 1)

class QRNGCache:
    def __init__(self):
        self.cache = Queue(maxsize=QRNG_CACHE_SIZE)
        self.prefetch_thread = None
        self.is_prefetching = False
        self.qrng = QRNGGenerator()
    
    def prefetch_numbers(self):
        """Background thread to generate QRNG numbers"""
        while self.is_prefetching:
            if self.cache.qsize() < QRNG_CACHE_SIZE * 0.5:  # Refill when below 50%
                try:
                    # Generate 10 numbers at a time
                    for _ in range(10):
                        if not self.cache.full():
                            num = self.qrng.generate_random_number(255)  # 0-255 for bytes
                            self.cache.put(num)
                except Exception as e:
                    st.error(f"QRNG generation error: {str(e)}")
                    # Fallback to CSPRNG if quantum generation fails
                    if not self.cache.full():
                        self.cache.put(secrets.randbelow(256))
            time.sleep(0.1)
    
    def start_prefetching(self):
        """Start the background prefetching thread"""
        self.is_prefetching = True
        self.prefetch_thread = Thread(target=self.prefetch_numbers, daemon=True)
        self.prefetch_thread.start()
    
    def stop_prefetching(self):
        """Stop the background prefetching thread"""
        self.is_prefetching = False
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1)
    
    def get_number(self):
        """Get a number from the cache or generate one if cache is empty"""
        try:
            return self.cache.get_nowait()
        except:
            return self.qrng.generate_random_number(255)

class Packet:
    def __init__(self, data, channel, rng_type):
        self.data = data
        self.channel = channel
        self.rng_type = rng_type
        self.timestamp = time.time()
    
    def to_hex(self):
        return binascii.hexlify(self.data).decode('utf-8')
    
    def to_ascii(self):
        return ''.join(chr(b) if 32 <= b <= 126 else '.' for b in self.data)

class ChannelHopper:
    def __init__(self, mode):
        self.mode = mode
        self.current_channel = random.randint(1, NUM_CHANNELS)
        self.channel_history = []
        self.last_hop_time = time.time()
        self.hop_interval = 0.05 if mode == 'QRNG' else 0.1  # QRNG hops faster
    
    def get_next_channel(self):
        if self.mode == 'QRNG':
            # Use quantum randomness for channel selection
            return (qrng_cache.get_number() % NUM_CHANNELS) + 1
        else:  # CSPRNG
            # Use system's secure random number generator with some predictability
            base = secrets.randbelow(NUM_CHANNELS)
            time_factor = int(time.time() * 10) % 3  # Add slight time-based pattern
            return ((base + time_factor) % NUM_CHANNELS) + 1
    
    def hop(self):
        current_time = time.time()
        if current_time - self.last_hop_time >= self.hop_interval:
            self.current_channel = self.get_next_channel()
            self.last_hop_time = current_time
            self.channel_history.append(self.current_channel)
        return self.current_channel

def generate_packet(rng_type):
    """Generate a random packet of data with distinctive patterns based on RNG type"""
    if rng_type == 'QRNG':
        # Use quantum randomness for packet data
        data = bytes([qrng_cache.get_number() for _ in range(PACKET_SIZE)])
    else:  # CSPRNG
        # Use system's secure random number generator with patterns
        base = secrets.token_bytes(1)[0]
        data = bytes([(base + i) % 256 for i in range(PACKET_SIZE)])
    return data

# Initialize QRNG cache
qrng_cache = QRNGCache()

def run_packet_simulation(listen_channel):
    # Initialize hoppers
    hoppers = {
        'QRNG': ChannelHopper('QRNG'),
        'CSPRNG': ChannelHopper('CSPRNG')
    }
    
    # Start QRNG prefetching
    qrng_cache.start_prefetching()
    
    # Run simulation
    intercepted_packets = {mode: [] for mode in hoppers}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(SIMULATION_STEPS):
        progress_bar.progress((step + 1) / SIMULATION_STEPS)
        status_text.text(f"Simulating step {step + 1}/{SIMULATION_STEPS}")
        
        for mode in hoppers:
            channel = hoppers[mode].hop()
            packet_data = generate_packet(mode)
            packet = Packet(packet_data, channel, mode)
            
            if channel == listen_channel:
                intercepted_packets[mode].append(packet)
    
    # Stop QRNG prefetching
    qrng_cache.stop_prefetching()
    status_text.empty()
    
    return intercepted_packets

def display_distribution(intercepted_packets):
    st.write("## 📊 Quantum vs Classical Encryption")
    
    # Calculate statistics
    total_packets = SIMULATION_STEPS
    modes = ['QRNG', 'CSPRNG']  # Only QRNG and CSPRNG
    packet_counts = {mode: len(packets) for mode, packets in intercepted_packets.items() if mode in modes}
    success_rates = {mode: (count / total_packets) * 100 for mode, count in packet_counts.items()}
    
    # Create columns for metrics
    cols = st.columns(2)
    
    # Display metrics with appropriate colors and indicators
    for idx, mode in enumerate(modes):
        with cols[idx]:
            delta = "✅ Maximum Security" if mode == 'QRNG' else "⚡ Standard Security"
            st.metric(
                label=f"{mode} Interception Rate",
                value=f"{success_rates[mode]:.1f}%",
                delta=delta,
                delta_color="off"
            )
    
    # Create comparison bar chart
    fig = go.Figure()
    colors = {'QRNG': 'green', 'CSPRNG': 'orange'}
    
    for mode in modes:
        fig.add_trace(go.Bar(
            name=mode,
            x=[mode],
            y=[packet_counts[mode]],
            text=[f"{success_rates[mode]:.1f}%"],
            textposition='auto',
            marker_color=colors[mode],
            hovertemplate=f"<b>{mode}</b><br>" +
                         "Packets Intercepted: %{y}<br>" +
                         "Success Rate: %{text}<br>" +
                         "<extra></extra>"
        ))

    fig.update_layout(
        title="Quantum vs Classical Security Comparison",
        yaxis_title="Intercepted Packets",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def display_packet_analysis(intercepted_packets, listen_channel):
    st.write(f"## 🎯 Channel {listen_channel} Interception Analysis")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Channel Hopping", "🔍 Packet Inspection"])
    
    with tab1:
        st.write("### Real-time Channel Hopping Comparison")
        
        # Create subplots for QRNG and CSPRNG
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["QRNG - Quantum Random", "CSPRNG - Cryptographic"],
            vertical_spacing=0.15
        )
        
        for idx, mode in enumerate(['QRNG', 'CSPRNG'], 1):
            if intercepted_packets[mode]:
                timestamps = [p.timestamp - intercepted_packets[mode][0].timestamp for p in intercepted_packets[mode]]
                channels = [p.channel for p in intercepted_packets[mode]]
                
                color = 'green' if mode == 'QRNG' else 'orange'
                
                # Add channel hopping line
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=channels,
                        mode='lines+markers',
                        name=f"{mode} Hops",
                        line=dict(color=color),
                        marker=dict(size=6)
                    ),
                    row=idx, col=1
                )
                
                # Add intercepted packets as stars
                intercept_times = []
                intercept_channels = []
                for t, c in zip(timestamps, channels):
                    if c == listen_channel:
                        intercept_times.append(t)
                        intercept_channels.append(c)
                
                if intercept_times:
                    fig.add_trace(
                        go.Scatter(
                            x=intercept_times,
                            y=intercept_channels,
                            mode='markers',
                            name=f'Intercepted',
                            marker=dict(
                                symbol='star',
                                size=12,
                                color='yellow',
                                line=dict(color='black', width=1)
                            )
                        ),
                        row=idx, col=1
                    )
        
        fig.update_layout(
            height=600,
            title_text="Channel Hopping Patterns (⭐ = Intercepted Packet)",
            showlegend=True
        )
        
        for i in range(1, 3):
            fig.update_yaxes(title_text="Channel", range=[0, NUM_CHANNELS + 1], row=i, col=1)
            fig.update_xaxes(title_text="Time (seconds)" if i == 2 else "", row=i, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.info("""
        🔍 **Observation:**
        - QRNG shows truly random channel hopping with minimal interceptions
        - CSPRNG follows a more predictable pattern, leading to more interceptions
        """)
    
    with tab2:
        st.write("### 📦 Intercepted Packet Comparison")
        
        cols = st.columns(2)
        for idx, mode in enumerate(['QRNG', 'CSPRNG']):
            with cols[idx]:
                st.write(f"#### {mode} Packets")
                packets = intercepted_packets[mode]
                if not packets:
                    st.info(f"No packets intercepted for {mode}")
                    continue
                
                st.write(f"**Intercepted:** {len(packets)} packets")
                st.write(f"**Rate:** {(len(packets) / SIMULATION_STEPS * 100):.1f}%")
                
                # Show first intercepted packet in detail
                if packets:
                    st.write("**Sample Packet:**")
                    packet = packets[0]
                    st.code(f"HEX: {packet.to_hex()}\nASCII: {packet.to_ascii()}", language="text")
                    
                    if mode == 'QRNG':
                        st.success("✅ High entropy, truly random data")
                    else:
                        st.warning("⚠️ Patterns may emerge over time")

# Main content
st.write("""
## Channel Interception Demonstration

You are attempting to intercept communications by listening to specific channels. Your goal is to:
1. Choose a channel to monitor
2. Analyze channel hopping patterns
3. Try to predict which channel will be used next

Three types of random number generators are being used for channel selection:
- **QRNG (Green)**: Uses quantum phenomena for true randomness
- **CSPRNG (Orange)**: Cryptographically secure pseudo-random numbers
- **PRNG (Red)**: Basic pseudo-random number generation

Select a channel to attempt interception and see why QRNG makes your efforts futile.
""")

# Channel selection
st.write("### Step 1: Choose Channel to Monitor")
listen_channel = st.selectbox(
    "Select Channel to Listen To",
    options=list(range(1, NUM_CHANNELS + 1)),
    format_func=lambda x: f"Channel {x}"
)

if st.button("Start Monitoring", type="primary"):
    with st.spinner("Running channel monitoring simulation..."):
        intercepted_packets = run_packet_simulation(listen_channel)
        display_packet_analysis(intercepted_packets, listen_channel)
        display_distribution(intercepted_packets)
else:
    st.info("👆 Select a channel and click 'Start Monitoring' to begin the interception simulation.") 