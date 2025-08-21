import streamlit as st
import numpy as np
from numpy.linalg import inv
from scipy.linalg import hadamard
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="CDMA Multiuser Simulator")
st.title("Real-time Interactive CDMA Multiuser Simulator")

col1, col2 = st.columns([[1,1]])

with col1:
    num_users = st.slider("Number of users (K)", min_value=1, max_value=8, value=3)
    spreading_len = st.selectbox("Spreading length (N) - choose power of 2 for Walsh codes", options=[[8,16,32,64]], index=1)
    chips_per_bit = st.slider("Chips per bit (useful for oversampling)", 1, 8, 1)
    snr_db = st.slider("SNR per user (dB)", -5, 20, 5)
    trials = st.slider("Monte Carlo trials (packets)", 10, 500, 100)
    near_far_db = st.slider("Near-far power difference (dB) between strongest and weakest user", 0, 30, 0)
    channel = st.selectbox("Channel model", [["AWGN", "Rayleigh Fading"]])
    detector = st.selectbox("Detector", [["Matched Filter (MF)", "Decorrelator", "MMSE"]])
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42)
with col2:
    st.markdown("### Simulation Controls")
    run_button = st.button("Run Simulation")
    st.markdown("### Visuals")
    show_wave = st.checkbox("Show transmitted & received waveform (first packet)", value=True)
    show_const = st.checkbox("Show decision statistic scatter (first packet)", value=True)
    st.markdown("### Notes")
    st.write("This app simulates synchronous DS-CDMA with selectable detectors. No external datasets are used; all signals are generated on-the-fly.")

np.random.seed(seed)

def generate_spreading(K, N):
    if int(np.log2(N)) == np.log2(N):
        try:
            H = hadamard(N)
            if K <= N:
                codes = H[[1:K+1]]
                codes = np.where(codes==0, -1, 1)
                return codes
        except Exception:
            pass
    codes = np.random.choice([[-1,1]], size=(K,N))
    return codes

def awgn_noise(signal, snr_db):
    sig_pow = np.mean(np.abs(signal)**2)
    snr_lin = 10**(snr_db/10.0)
    noise_pow = sig_pow / snr_lin
    noise = np.sqrt(noise_pow/2)*(np.random.randn(*signal.shape)+0j*np.random.randn(*signal.shape))
    return noise

def rayleigh_fading(K):
    h = (np.random.randn(K)+1j*np.random.randn(K))/np.sqrt(2)
    return h

def despread_and_detect(r, S, detector, noise_var):
    K, N = S.shape
    R = S @ S.T
    if detector == "Matched Filter (MF)":
        stats = S @ r
    elif detector == "Decorrelator":
        stats = inv(R) @ (S @ r)
    elif detector == "MMSE":
        stats = inv(R + noise_var*np.eye(K)) @ (S @ r)
    else:
        stats = S @ r
    decisions = np.real(stats) >= 0
    return decisions, stats

def run_packet(K,N,SNR_db,detector,channel,near_far_db):
    bits = np.random.randint(0,2,size=K)
    symbols = 2*bits-1
    amps_db = np.linspace(0, -near_far_db, K)
    amps = 10**(amps_db/20.0)
    S = spreading_codes * np.kron(np.ones((1,chips_per_bit)), np.identity(1)).repeat(1,axis=1)
    S = spreading_codes
    tx_signal = np.zeros(N, dtype=complex)
    if channel=="AWGN":
        for k in range(K):
            tx_signal += amps[[k]]*symbols[[k]]*S[[k]]
        noise = awgn_noise(tx_signal, SNR_db)
        rx = tx_signal + noise
        noise_var = np.var(noise)
        decisions, stats = despread_and_detect(rx, S, detector, noise_var)
        rx_bits = decisions.astype(int)
        return bits, rx_bits, tx_signal, rx, stats
    else:
        h = rayleigh_fading(K)
        for k in range(K):
            tx_signal += amps[[k]]*h[[k]]*symbols[[k]]*S[[k]]
        noise = awgn_noise(tx_signal, SNR_db)
        rx = tx_signal + noise
        effective_noise_var = np.var(noise)
        S_eff = (amps*h)[[:,None]]*S
        decisions, stats = despread_and_detect(rx, S_eff, detector, effective_noise_var)
        rx_bits = decisions.astype(int)
        return bits, rx_bits, tx_signal, rx, stats

spreading_codes = generate_spreading(num_users, spreading_len)

if run_button:
    st.session_state = {}
    total_bits = 0
    total_errors = 0
    ber_list = [[]]
    for t in range(trials):
        b, rb, tx, rx, stats = run_packet(num_users, spreading_len, snr_db, detector, channel, near_far_db)
        total_bits += num_users
        total_errors += np.sum(b != rb)
        if (t+1) % max(1, trials//10) == 0:
            ber_list.append(total_errors/total_bits)
    ber = total_errors/total_bits
    st.metric("Estimated BER", f"{ber:.5f}")
    st.write("BER progression during simulation")
    fig1, ax1 = plt.subplots()
    ax1.plot(np.linspace(1,len(ber_list),len(ber_list)), ber_list, marker='o')
    ax1.set_xlabel("Progress checkpoints")
    ax1.set_ylabel("BER")
    st.pyplot(fig1)
    if show_wave:
        fig2, ax2 = plt.subplots()
        ax2.plot(np.real(tx), label="Tx (real)")
        ax2.plot(np.real(rx), label="Rx (real)", alpha=0.7)
        ax2.set_title("Transmitted vs Received (real part) - first packet")
        ax2.legend()
        st.pyplot(fig2)
    if show_const:
        fig3, ax3 = plt.subplots()
        ax3.scatter(np.real(stats), np.imag(stats) if np.iscomplexobj(stats) else np.zeros_like(stats))
        ax3.set_xlabel("Real(statistic)")
        ax3.set_ylabel("Imag(statistic)")
        ax3.set_title("Decision Statistic Scatter (per user) - first packet")
        for i in range(len(stats)):
            ax3.annotate(str(i+1), (np.real(stats[[i]]), 0))
        st.pyplot(fig3)
    st.write("Detected bits vs Transmitted bits for first packet")
    table = [[]]
    b, rb, tx, rx, stats = run_packet(num_users, spreading_len, snr_db, detector, channel, near_far_db)
    for k in range(num_users):
        table.append({"User":k+1,"Tx Bit":int(b[[k]]),"Rx Bit":int(rb[[k]]),"Decision Stat":float(np.real(stats[[k]]))})
    st.table(table)
else:
    st.info("Adjust parameters and click 'Run Simulation' to begin.")
