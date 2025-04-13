import streamlit as st
import numpy as np
import pandas as pd
import time
import requests
import random
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from functools import lru_cache
from threading import Thread
from queue import Queue

# Configure page
st.set_page_config(page_title="QRNG vs Classical RNG Comparison", layout="wide")
st.title("Quantum vs Classical Random Number Generation in Frequency Hopping")

# Constants
NUM_CHANNELS = 13
SIMULATION_STEPS = 50
QRNG_CACHE_SIZE = 100

# Sidebar for jamming configuration
st.sidebar.title("Jamming Configuration")

# Let user select channels to jam
st.sidebar.subheader("Select Channels to Jam")
available_channels = list(range(1, NUM_CHANNELS + 1))
selected_channels = st.sidebar.multiselect(
    "Choose channels to jam (1-13):",
    available_channels,
    default=[5, 6]  # Default selection
)

# Let user configure jamming periods
st.sidebar.subheader("Jamming Period")
jam_start = st.sidebar.slider("Start jamming at step:", 0, SIMULATION_STEPS-1, 10)
jam_duration = st.sidebar.slider("Jamming duration (steps):", 1, SIMULATION_STEPS-jam_start, 5)

# Create jamming scenario from user input
JAMMING_SCENARIOS = {
    "User Defined": {(jam_start, jam_start + jam_duration): selected_channels}
}

# Display selected jamming configuration
st.sidebar.subheader("Current Jamming Setup")
st.sidebar.write(f"Jamming channels {selected_channels}")
st.sidebar.write(f"From step {jam_start} to {jam_start + jam_duration}")

class Interceptor:
    def __init__(self):
        self.predicted_channels = []
        self.successful_interceptions = 0
        self.prediction_accuracy = 0
        self.last_observed_channel = None
        self.pattern_history = []
    
    def attempt_prediction(self, current_channel, rng_type):
        """Attempt to predict the next channel based on observed patterns"""
        self.pattern_history.append(current_channel)
        if len(self.pattern_history) > 5:
            self.pattern_history.pop(0)
        
        if rng_type == 'PRNG':
            # PRNG is most predictable
            if len(self.pattern_history) >= 3:
                # Try to detect simple patterns
                if self.pattern_history[-3:] == [self.pattern_history[-1]] * 3:
                    predicted = (current_channel + 1) % NUM_CHANNELS
                else:
                    predicted = (current_channel + random.randint(1, 3)) % NUM_CHANNELS
            else:
                predicted = (current_channel + 1) % NUM_CHANNELS
        elif rng_type == 'CSPRNG':
            # CSPRNG is moderately predictable
            if len(self.pattern_history) >= 5:
                # Try to detect more complex patterns
                predicted = (current_channel + random.randint(1, 5)) % NUM_CHANNELS
            else:
                predicted = (current_channel + 2) % NUM_CHANNELS
        else:  # QRNG
            # QRNG is virtually unpredictable
            predicted = random.randint(1, NUM_CHANNELS)
        
        self.predicted_channels.append(predicted)
        return predicted

def run_simulation(jamming_scenario):
    # Start QRNG prefetching
    qrng_cache.start_prefetching()
    
    results = {
        'QRNG': {'channels': [], 'latencies': [], 'collisions': 0},
        'CSPRNG': {'channels': [], 'latencies': [], 'collisions': 0},
        'PRNG': {'channels': [], 'latencies': [], 'collisions': 0}
    }
    
    # Initialize interceptors for each RNG type
    interceptors = {
        'QRNG': Interceptor(),
        'CSPRNG': Interceptor(),
        'PRNG': Interceptor()
    }
    
    hoppers = {mode: ChannelHopper(mode) for mode in ['QRNG', 'CSPRNG', 'PRNG']}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(SIMULATION_STEPS):
        progress_bar.progress((step + 1) / SIMULATION_STEPS)
        status_text.text(f"Simulating step {step + 1}/{SIMULATION_STEPS}")
        
        jammed_channels = set()
        for (start, end), channels in JAMMING_SCENARIOS[jamming_scenario].items():
            if start <= step <= end:
                jammed_channels.update(channels)
        
        for mode in ['QRNG', 'CSPRNG', 'PRNG']:
            channel, latency = hoppers[mode].get_next_channel(jammed_channels)
            results[mode]['channels'].append(channel)
            results[mode]['latencies'].append(latency)
            
            # Let interceptor attempt to predict
            predicted = interceptors[mode].attempt_prediction(channel, mode)
            
            # Count successful interceptions
            if predicted == channel:
                interceptors[mode].successful_interceptions += 1
            
            if channel in jammed_channels:
                results[mode]['collisions'] += 1
    
    # Stop QRNG prefetching
    qrng_cache.stop_prefetching()
    
    # Calculate interception success rates
    for mode in interceptors:
        interceptors[mode].prediction_accuracy = (
            interceptors[mode].successful_interceptions / SIMULATION_STEPS * 100
        )
    
    status_text.empty()
    progress_bar.empty()
    return results, interceptors

def create_visualization(results, interceptors, jamming_scenario):
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'Channel Hopping Patterns',
            'Latency Over Time',
            'Collision Count',
            'Interceptor Success Rate'
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Colors for each mode
    colors = {'QRNG': '#00ff00', 'CSPRNG': '#ffa500', 'PRNG': '#ff0000'}
    
    # Channel hopping visualization
    for mode in results:
        # Actual channels
        fig.add_trace(
            go.Scatter(
                x=list(range(SIMULATION_STEPS)),
                y=results[mode]['channels'],
                name=f"{mode} Channels",
                mode='lines+markers',
                line=dict(color=colors[mode]),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Predicted channels
        fig.add_trace(
            go.Scatter(
                x=list(range(SIMULATION_STEPS)),
                y=interceptors[mode].predicted_channels,
                name=f"{mode} Predicted",
                mode='markers',
                marker=dict(
                    color=colors[mode],
                    symbol='x',
                    size=8
                ),
                showlegend=True
            ),
            row=1, col=1
        )
    
    # Latency visualization
    for mode in results:
        fig.add_trace(
            go.Scatter(
                x=list(range(SIMULATION_STEPS)),
                y=results[mode]['latencies'],
                name=f"{mode} Latency",
                line=dict(color=colors[mode]),
                showlegend=True
            ),
            row=2, col=1
        )
    
    # Collision count bar chart
    fig.add_trace(
        go.Bar(
            x=list(results.keys()),
            y=[results[mode]['collisions'] for mode in results],
            marker_color=[colors[mode] for mode in results],
            name='Collisions'
        ),
        row=3, col=1
    )
    
    # Interception success rate
    fig.add_trace(
        go.Bar(
            x=list(interceptors.keys()),
            y=[interceptors[mode].prediction_accuracy for mode in interceptors],
            marker_color=[colors[mode] for mode in interceptors],
            name='Prediction Accuracy'
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"RNG Comparison under {jamming_scenario}",
        showlegend=True,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_yaxes(title_text="Channel", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Prediction Accuracy (%)", row=4, col=1)
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    
    return fig

# Main simulation control
st.header("Frequency Hopping Simulation with Interceptor")
st.write("""
This simulation demonstrates how different RNG methods perform under interception attempts:
- QRNG (Green): Quantum Random Number Generator - Virtually unpredictable
- CSPRNG (Orange): Cryptographically Secure Pseudo-Random Number Generator - Moderately predictable
- PRNG (Red): Basic Pseudo-Random Number Generator - Highly predictable

The 'X' markers show where the interceptor predicted the next channel would be.
""")

if st.button("Run Simulation", type="primary"):
    if not selected_channels:
        st.warning("Please select at least one channel to jam in the sidebar.")
    else:
        with st.spinner("Running simulation..."):
            results, interceptors = run_simulation("User Defined")
            
            # Display results
            st.plotly_chart(create_visualization(results, interceptors, "Jamming Simulation"), use_container_width=True)
            
            # Display statistics
            st.subheader("Performance Statistics")
            stats_cols = st.columns(3)
            
            for idx, mode in enumerate(['QRNG', 'CSPRNG', 'PRNG']):
                with stats_cols[idx]:
                    st.metric(
                        label=f"{mode} Statistics",
                        value=f"{results[mode]['collisions']} collisions",
                        delta=f"{np.mean(results[mode]['latencies']):.1f}ms avg latency"
                    )
            
            # Analysis
            st.subheader("Impact Analysis")
            qrng_advantage = (results['PRNG']['collisions'] - results['QRNG']['collisions']) / max(1, results['PRNG']['collisions']) * 100
            
            # Calculate average latencies
            avg_latencies = {mode: np.mean(results[mode]['latencies']) for mode in results}
            
            st.write(f"""
            ### Collision Analysis
            - QRNG showed {qrng_advantage:.1f}% fewer collisions compared to PRNG
            - QRNG collisions: {results['QRNG']['collisions']}
            - CSPRNG collisions: {results['CSPRNG']['collisions']}
            - PRNG collisions: {results['PRNG']['collisions']}
            
            ### Latency Analysis
            - QRNG average latency: {avg_latencies['QRNG']:.1f}ms
            - CSPRNG average latency: {avg_latencies['CSPRNG']:.1f}ms
            - PRNG average latency: {avg_latencies['PRNG']:.1f}ms
            
            ### Interception Analysis
            - QRNG prediction accuracy: {interceptors['QRNG'].prediction_accuracy:.1f}%
            - CSPRNG prediction accuracy: {interceptors['CSPRNG'].prediction_accuracy:.1f}%
            - PRNG prediction accuracy: {interceptors['PRNG'].prediction_accuracy:.1f}%
            
            ### Key Findings
            - {'✅' if results['QRNG']['collisions'] < results['PRNG']['collisions'] else '❌'} QRNG demonstrates superior jamming avoidance
            - {'✅' if avg_latencies['QRNG'] < avg_latencies['CSPRNG'] else '❌'} QRNG shows better latency performance
            - {'✅' if interceptors['QRNG'].prediction_accuracy < interceptors['PRNG'].prediction_accuracy else '❌'} QRNG is virtually impossible to predict
            """)