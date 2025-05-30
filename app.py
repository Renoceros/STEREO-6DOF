import streamlit as st

pages = {
    "Navigation": [
        st.Page("streamlit/Home.py", title="Home"),
        st.Page("streamlit/Implementation.py", title="Implementation"),
        st.Page("streamlit/Dashboard.py", title="Dashboard"),
        st.Page("streamlit/Documentation.py", title="Documentation")
    ]
}
# st.page_link("streamlit/Home.py", title="Home")
# st.page_link("streamlit/Implementation.py", title="Implementation")
# st.page_link("streamlit/Dashboard.py", title="Dashboard")
# st.page_link("streamlit/Documentation.py", title="Documentation")
# Show as top nav instead of sidebar
pg = st.navigation(pages, position="top")  # or use "sidebar" if preferred
pg.run()