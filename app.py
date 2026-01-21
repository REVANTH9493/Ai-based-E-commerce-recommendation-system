import streamlit as st


st.title("*AI enabled* **recommendation system**")
st.markdown("# Markdown Heading")

st.caption('this is hello')

st.header("recommended products based on user preference")
st.subheader("it is using the content and collaborative based filtering approaches")

st.text("*This is a best recommendation system*")
st.markdown("**hello**")
st.markdown(" - Bullet point 1\n - Bullet point 2")
st.divider()
name = st.text_input("Enter your name:", placeholder="John Doe")
st.write(f"Hello, {name}!")
password = st.text_input("Password:", type="password")

st.selectbox("recommended approach:",['content', 'collaborative', 'hybrid'])
hobbies = st.multiselect(
    "Select hobbies:",
    ["Reading", "Gaming", "Sports", "Music", "Coding"]
)

gender = st.radio("Gender:", ["Male", "Female", "Other"])
# Primary button (highlighted)
if st.button("Submit", type="primary"):
    st.balloons()

st.text_area("write your product name below")
if st.button('Click here'):
    st.write('hello')
data = 'this is button'
st.download_button(
    label="Download Data",
    data=data,
    file_name="data.txt"
)