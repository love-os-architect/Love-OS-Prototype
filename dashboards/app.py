import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from runner_pg import run_episode_with_state
from public_goods_env import PGConfig

st.set_page_config(layout="wide", page_title="Love-OS: Public Goods Dashboard")

st.title("Love-OS: Multi-Agent Public Goods Simulation")
st.markdown("""
**Observation-based Love/Ego Agents** with **Philosophy Modules** & **CEM Planning**.
Explore how **Love (Integration)**, **Philosophy**, and **Externality** affect the sustainability of the world.
""")

st.sidebar.header("Environment")
grid_size = st.sidebar.slider("Grid size", 8, 20, 12)
n_agents = st.sidebar.slider("# Agents", 2, 20, 6)
steps = st.sidebar.slider("Steps", 50, 500, 200)
base_regrow = st.sidebar.slider("Base regrow prob", 0.01, 0.1, 0.04, 0.01)
pub_mult = st.sidebar.slider("Public multiplier", 0.0, 2.0, 0.6, 0.1)
overharvest_p = st.sidebar.slider("Overharvest penalty", 0.1, 1.0, 0.5, 0.1)
donate_unit = st.sidebar.slider("Donate unit", 0.0, 0.5, 0.2, 0.05)
seed = st.sidebar.number_input("Seed", value=42)

st.sidebar.markdown("---")
st.sidebar.header("Agent & Love")
alpha_love = st.sidebar.slider("Alpha (Love)", 0.0, 5.0, 1.0, 0.1)
gamma_ego = st.sidebar.slider("Gamma (Ego)", 0.0, 2.0, 0.1, 0.05)
lambda_H = st.sidebar.slider("Lambda H (Harmony)", 0.0, 2.0, 0.3, 0.1)
lambda_X = st.sidebar.slider("Lambda X (Externality)", 0.0, 2.0, 0.3, 0.1)
lambda_I = st.sidebar.slider("Lambda I (InfoGain)", 0.0, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("Philosophy Module")
phil_mode = st.sidebar.selectbox("Mode", ["none", "buddhism", "stoic", "existential"])
phil_params = {}
agent_kwargs = {'alpha': alpha_love, 'gamma': gamma_ego, 'lambda_H': lambda_H, 'lambda_X': lambda_X, 'lambda_I': lambda_I}

if phil_mode == "buddhism":
    st.sidebar.subheader("Buddhism Params")
    l_B = st.sidebar.slider("Lambda B (Weight)", 0.0, 2.0, 0.5)
    eta1 = st.sidebar.slider("Eta1 (Stress)", 0.0, 2.0, 1.0)
    eta3 = st.sidebar.slider("Eta3 (Attachment)", 0.0, 2.0, 0.5)
    thr = st.sidebar.slider("Attach Threshold", 0.1, 2.0, 0.8)
    phil_params = {'eta1': eta1, 'eta3': eta3, 'attach_thr': thr}
    agent_kwargs['lambda_B'] = l_B
elif phil_mode == "stoic":
    st.sidebar.subheader("Stoic Params")
    l_S = st.sidebar.slider("Lambda S (Weight)", 0.0, 2.0, 0.5)
    zeta = st.sidebar.slider("Zeta (Uncontrollable)", 0.0, 2.0, 0.5)
    phil_params = {'zeta': zeta}
    agent_kwargs['lambda_S'] = l_S
elif phil_mode == "existential":
    st.sidebar.subheader("Existential Params")
    l_E = st.sidebar.slider("Lambda E (Consistency)", 0.0, 2.0, 0.3)
    agent_kwargs['lambda_E'] = l_E

st.sidebar.markdown("---")
st.sidebar.header("Planner")
planner_type = st.sidebar.selectbox("Type", ["1-step", "cem"])
planner_params = {}
if planner_type == "cem":
    cem_H = st.sidebar.slider("Horizon (H)", 2, 20, 5)
    cem_N = st.sidebar.slider("Samples (N)", 16, 128, 64)
    cem_iters = st.sidebar.slider("Iters", 1, 5, 3)
    elite_frac = st.sidebar.slider("Elite Frac", 0.05, 0.5, 0.2)
    other_pol = st.sidebar.selectbox("Others' Policy", ["stay", "random", "mean_field"])
    planner_params = {'H': cem_H, 'N': cem_N, 'iters': cem_iters, 'elite_frac': elite_frac, 'other_policy': other_pol}
else:
    planner_type = None

if st.sidebar.button("Run Simulation"):
    cfg = PGConfig(size=grid_size, n_agents=n_agents, seed=seed, base_regrow_p=base_regrow, public_multiplier=pub_mult, overharvest_penalty=overharvest_p, donate_unit=donate_unit)
    with st.spinner("Running simulation..."):
        logs, final_state = run_episode_with_state(steps=steps, cfg=cfg, learn=True, philosophy_mode=phil_mode, philosophy_params=phil_params, agent_kwargs=agent_kwargs, planner=planner_type, planner_params=planner_params)
    
    st.subheader("Metrics Time Series")
    fig1, ax1 = plt.subplots(1, 4, figsize=(20, 4))
    ax1[0].plot(logs['F'], label='Free Energy'); ax1[0].set_title("Free Energy (Stress)"); ax1[0].legend()
    ax1[1].plot(logs['valence'], color='orange', label='Valence'); ax1[1].set_title("Valence (Emotional)"); ax1[1].legend()
    ax1[2].plot(logs['harmony'], color='green', label='Harmony'); ax1[2].set_title("Social Harmony (Cos Sim)"); ax1[2].legend()
    ax1[3].plot(logs['resource_mean'], label='Mean Resource'); ax1[3].plot(logs['public_pool'], label='Public Pool', linestyle='--'); ax1[3].set_title("Resources"); ax1[3].legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(1, 4, figsize=(20, 4))
    ax2[0].plot(logs['ext_net'], label='Net'); ax2[0].plot(logs['ext_benefit'], label='Benefit (Donate)', alpha=0.5); ax2[0].plot(logs['ext_harm'], label='Harm (Harvest)', alpha=0.5); ax2[0].set_title("Externality (Karma)"); ax2[0].legend()
    ax2[1].plot(logs['D_B'], color='purple', label='D_B (Suffering)'); ax2[1].set_title("Buddhism: Suffering")
    ax2[2].plot(logs['C_Stoic'], color='brown', label='C_Stoic'); ax2[2].set_title("Stoic: Disturbance")
    ax2[3].plot(logs['Resp_KL'], color='red', label='Resp_KL'); ax2[3].set_title("Existential: Inconsistency")
    st.pyplot(fig2)

    st.subheader("Final State")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Resource Grid (Darker = Richer)")
        fig3, ax3 = plt.subplots()
        im = ax3.imshow(final_state['grid'], cmap='Greens', vmin=0, vmax=1)
        pos = final_state['pos']
        ax3.scatter(pos[:,1], pos[:,0], c='red', s=50, label='Agents')
        plt.colorbar(im, ax=ax3); ax3.legend()
        st.pyplot(fig3)
    with col2:
        st.caption("Summary Stats")
        st.write(f"**Mean F:** {np.mean(logs['F']):.4f}")
        st.write(f"**Mean Valence:** {np.mean(logs['valence']):.4f}")
        st.write(f"**Final Harmony:** {logs['harmony'][-1]:.4f}")
        st.write(f"**Final Public Pool:** {logs['public_pool'][-1]:.4f}")
        st.write(f"**Avg Net Externality:** {np.mean(logs['ext_net']):.4f}")
