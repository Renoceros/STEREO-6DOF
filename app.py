import streamlit as st

pages = {
    "Navigation": [
        st.Page("streamlit/Home.py", title="Home"),
        st.Page("streamlit/Implementation.py", title="Implementation"),
        st.Page("streamlit/Dashboard.py", title="Dashboard"),
        st.Page("streamlit/Documentation.py", title="Documentation"),
        st.Page("streamlit/Train.py", title="Train a Model")
    ]
}

# Show as top nav instead of sidebar
pg = st.navigation(pages, position="top")  # or use "sidebar"
pg.run()