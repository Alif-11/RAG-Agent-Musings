import streamlit as st

st.title("huh?")
st.write("yo")

api_key = ""

with st.sidebar:
  st.header("HUH?")
  api_key = st.text_input("Yo???? Input???", type="password", help="Yo what's up?")
  

print(f"bruh {api_key}")