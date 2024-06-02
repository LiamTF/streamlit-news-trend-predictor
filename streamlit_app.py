import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import time
from TrendProcesses import FetchData, CreateFeatures, RunAnalysis, RunModels


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='The News Trend Predictor',
    page_icon=':newspaper:',
    initial_sidebar_state='collapsed',
)

# CSS:
with open(Path(__file__).parent/'style.css') as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

def set_bg_hack(main_bg):
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .appview-container {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
#set_bg_hack('desk-blurred.jpg')



# Initialization of session state:
if 'item_string' not in st.session_state:
    st.session_state.item_string = ''
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def restart():
    st.session_state.submitted = False
    st.rerun()


def submit_button(new_item_string):
    string = new_item_string.lower().strip()
    # Check errors:
    if st.session_state.submitted == True:
        return
    if string == '':
        st.warning('Please enter a news item string!', icon='🚨')
    else:
        st.session_state.item_string = string
        st.session_state.submitted = True
    return

def plot_data(df_DT, df_KNN, df_LR, df):

    st.header(f'Report: \'{st.session_state.item_string}\'')
    
    ''
    ''

    st.subheader(':deciduous_tree: Decision Tree Model')
    st.line_chart(
        df_DT,
        x=None,
        y=('real', 'prediction'),
        color=('#43b828', '#03a1fc')
    )

    ''
    ''

    st.subheader(':runner: K-Nearest Neighbour Model')
    st.line_chart(
        df_KNN,
        x=None,
        y=('real', 'prediction'),
        color=('#de12e6', '#03a1fc')
    )

    ''
    ''

    st.subheader(':wrench: Linear Regression Model')
    st.line_chart(
        df_LR,
        x=None,
        y=('real', 'prediction'),
        color=('#e61247', '#03a1fc')
    )

    return

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.markdown(
        f"""
        <h1>
        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" class="svgicon" fill="#5f6368"><path d="m105-233-65-47 200-320 120 140 160-260 109 163q-23 1-43.5 5.5T545-539l-22-33-152 247-121-141-145 233ZM863-40 738-165q-20 14-44.5 21t-50.5 7q-75 0-127.5-52.5T463-317q0-75 52.5-127.5T643-497q75 0 127.5 52.5T823-317q0 26-7 50.5T795-221L920-97l-57 57ZM643-217q42 0 71-29t29-71q0-42-29-71t-71-29q-42 0-71 29t-29 71q0 42 29 71t71 29Zm89-320q-19-8-39.5-13t-42.5-6l205-324 65 47-188 296Z"/></svg>
        The News Trend Predictor
        </h1>
        """,
        unsafe_allow_html=True
    )


if st.session_state.submitted == False:
    st.header('About')
    st.markdown("""
        Our Introduction here!
        Our Introduction here!
        Our Introduction here!
        Our Introduction here!
        Our Introduction here!
    """, unsafe_allow_html=True)


    st.header('Generate a Report')
    item_input = st.text_input(label='Enter your news item search string')
    st.button(label='Generate', on_click=submit_button, args=[item_input])
    ''
    st.markdown("""
        :warning:
        Report generation can take up to 3 minutes.
        <br>
        :warning:
        We highly recommend avoiding unnecessary characters and spaces, and using broad search strings.
    """, unsafe_allow_html=True)
else:

    if st.button(label='Reset'):
        restart()
    st.markdown("""
        <i style="max-width: 50%;">Note: Resetting will remove the current report from memory.</i>
    """, unsafe_allow_html=True)



    raw_data_loader = st.progress(0, text=f"Fetching trend data for {st.session_state.item_string}")
    for percent_complete in range(100):
        raw_data_loader.progress(percent_complete + 1, text=f"Fetching trend data for '{st.session_state.item_string}'.")
    time.sleep(1)
    raw_data_loader.empty()
    data_fetcher = FetchData(raw_data_loader)
    trend, yt_data = data_fetcher.fetch_and_return_final_df_list(st.session_state.item_string)

    # TESTING DATA:
    #all_trends = pd.read_csv(Path(__file__).parent/'data/test_trends_data.csv')
    #trend = all_trends.loc[:, ["date", all_trends.columns[1]]]
    #yt_data = pd.read_csv(Path(__file__).parent/'data/test_yt_data.csv')

    remaining_processes_loader = st.progress(0, text=f"Feature engineering datasets for '{st.session_state.item_string}'.")
    feature_creator = CreateFeatures()
    data, data_normalised = feature_creator.create_features(trend, yt_data)

    remaining_processes_loader.progress(33, text=f"Analysing datasets for '{st.session_state.item_string}'.")
    # TODO - add analysis and plot analysis:
    #corr_matrix = analyser = RunAnalysis()

    remaining_processes_loader.progress(66, text=f"Modelling predictions for '{st.session_state.item_string}'.")
    modeller = RunModels()
    df_DT, df_KNN, df_LR = modeller.run_all_models(data_normalised)

    remaining_processes_loader.progress(99, text=f"Plotting results for '{st.session_state.item_string}'.")


    remaining_processes_loader.empty()
    plot_data(df_DT, df_KNN, df_LR, data)

''
''
