import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def show_sidebar_logo():
    logo = "https://media.discordapp.net/attachments/1336319606377287802/1506548787873714266/image.png?ex=6a0eaa65&is=6a0d58e5&hm=b99afb38f0dc4cdad119ea6b23dbf8ab7354e8b2a32530c7f314b2cd02eabb70&=&format=webp&quality=lossless&width=576&height=587"

    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="{logo}" width="120"
                 style="border-radius: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
        </div>
        """,
        unsafe_allow_html=True
    )


def show_prediction_result(real_prob, fake_prob, label, risk):
    st.divider()
    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Real Probability", f"{real_prob:.4f}")

    with c2:
        st.metric("Fake Probability", f"{fake_prob:.4f}")

    with c3:
        st.metric("Decision", label)

    st.progress(real_prob)

    st.write("### Interpretation")
    st.write(f"**Result:** {label}")
    st.write(f"**Risk Level:** {risk}")


def show_benford_result(benford_div, benford_label, benford_explanation):
    st.write("### Forensic Feature Analysis")

    b1, b2 = st.columns(2)

    with b1:
        if benford_div is not None:
            st.metric("Benford DCT Divergence", f"{benford_div:.6f}")
        else:
            st.metric("Benford DCT Divergence", "N/A")

    with b2:
        st.metric("Benford Risk Level", benford_label)

    st.write(f"**Benford Explanation:** {benford_explanation}")


def show_benford_chart(actual_dist, benford_dist):
    if actual_dist is None or benford_dist is None:
        return

    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    chart_mode = st.radio(
        "Benford Chart Display",
        [
            "Observed vs Benford's Law",
            "Observed only",
            "Benford's Law only"
        ],
        horizontal=True
    )

    digits = np.arange(1, 10)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    def add_percentage_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f"{height * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

    if chart_mode == "Observed vs Benford's Law":
        # Benford's Law on the left
        bars1 = ax.bar(
            digits - width / 2,
            benford_dist,
            width,
            label="Benford's Law"
        )

        # Observed on the right
        bars2 = ax.bar(
            digits + width / 2,
            actual_dist,
            width,
            label="Observed"
        )

        add_percentage_labels(bars1)
        add_percentage_labels(bars2)

        max_y = max(max(actual_dist), max(benford_dist))

    elif chart_mode == "Observed only":
        bars = ax.bar(
            digits,
            actual_dist,
            width,
            label="Observed"
        )

        add_percentage_labels(bars)
        max_y = max(actual_dist)

    else:
        bars = ax.bar(
            digits,
            benford_dist,
            width,
            label="Benford's Law"
        )

        add_percentage_labels(bars)
        max_y = max(benford_dist)

    ax.set_title("First Digit Distribution")
    ax.set_xlabel("First Digit")
    ax.set_ylabel("Proportion")
    ax.set_xticks(digits)
    ax.set_ylim(0, max_y + 0.08)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)