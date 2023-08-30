import streamlit as st
import streamlit.components.v1 as components


st.header("Competition Info")
st.subheader("Description")
st.write(
    """
    In the year 2912, the Spaceship Titanic collided with a spacetime anomaly and almost half of the passengers
    were transported to an alternate dimension. To help retrieve the lost passengers,
    you are challenged to predict which passengers were transported using records recovered 
    from the spaceship's computer system.

    So, it is a binary classification problem.
    """
)
st.subheader("Evaluation")
st.write(
    """
    Submissions are evaluated based on their **classification accuracy**,
    the percentage of predicted labels that are correct.
    """
)

st.subheader("What have we done:")
components.iframe('https://docs.google.com/presentation/d/e/2PACX-1vSPt_mpZ9cXdEh1AVxwMhuRpwlaPeqxuLaOkaoVHr0kpdLkLAoS2Bbq0aUGkAFqlwudVJ3fmtYUHeK2/embed?start=false&loop=false&delayms=5000',height=565)
