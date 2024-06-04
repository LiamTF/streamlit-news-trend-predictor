import streamlit as st
import pandas as pd
from pathlib import Path
import base64
import time
from TrendProcesses import FetchData, CreateFeatures, RunAnalysis, RunModels
import plotly.express as px


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='The News Trend Predictor',
    page_icon=':newspaper:',
    initial_sidebar_state='collapsed',
)

# CSS:
with open(Path(__file__).parent/'style.css') as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)



# Initialization of session state:
if 'item_string' not in st.session_state:
    st.session_state.item_string = ''
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# -----------------------------------------------------------------------------
# Declare some useful functions.

#@st.cache_data
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

def plot_data(df_LGBM, df_RF, df_KNN, df_LR, scores, corr_df, df):

    st.header(f'Report: \'{st.session_state.item_string}\'')
    ''
    ''
    st.subheader(':evergreen_tree::evergreen_tree: Random Forest Model')
    st.line_chart(
        df_RF,
        x=None,
        y=('real', 'prediction'),
        color=('#3e8f2b', '#03a1fc')
    )
    st.markdown(
        f"""
        <u>Mean Square Error</u><br>
        <b>{scores["randomforest"]}</b>
        """,
        unsafe_allow_html=True
    )
    ''
    ''
    st.subheader(':bulb: LightGBM Model')
    st.line_chart(
        df_LGBM,
        x=None,
        y=('real', 'prediction'),
        color=('#fff70a', '#03a1fc')
    )
    st.markdown(
        f"""
        <u>Mean Square Error</u><br>
        <b>{scores["lightgbm"]}</b>
        """,
        unsafe_allow_html=True
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
    st.markdown(
        f"""
        <u>Mean Square Error</u><br>
        <b>{scores["knn"]}</b>
        """,
        unsafe_allow_html=True
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
    st.markdown(
        f"""
        <u>Mean Square Error</u><br>
        <b>{scores["linearregression"]}</b>
        """,
        unsafe_allow_html=True
    )
    ''

    fig = px.bar(corr_df, x='Feature', y='Correlation', title='Feature Correlation to Google Trend')
    st.plotly_chart(fig)

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
        <u>The News Trend Predictor</u> predicts how a given news item's Google Search popularity will trend for the rest of today.<br>
        <br>
        Here's how it works:        
        <ol>
            <li>Enter a news item's key search string.</li>
            <li>We fetch the last month of Google Trend data, and fetch & feature engineer popularity<br>
                features - we user data from the top 3 videos on the news item in the last 30 days, and incorporate calendar features.</li>
            <li>A selection of models are trained and tested on this data.</li>
            <li>The data from today's top 3 videos are extrapolated, and we use these features to predict the trend in populatity for the rest of today.</li>
            <li>Model accuracy is plotted along with today's forecast, so you can determine whether the model is effective when predicting your submitted new item.</li>
        </ol> 
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
    #for percent_complete in range(100):
    #    raw_data_loader.progress(percent_complete + 1, text=f"Fetching trend data for '{st.session_state.item_string}'.")
    #time.sleep(1)
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
    analyser = RunAnalysis()
    corr_matrix = analyser.get_corr_matrix(data_normalised)

    remaining_processes_loader.progress(66, text=f"Modelling predictions for '{st.session_state.item_string}'.")
    modeller = RunModels()
    df_LGBM, df_RF, df_KNN, df_LR, scores = modeller.run_all_models(data_normalised, corr_matrix)
    
    remaining_processes_loader.progress(99, text=f"Plotting results for '{st.session_state.item_string}'.")

    remaining_processes_loader.empty()
    plot_data(df_LGBM, df_RF, df_KNN, df_LR, scores, corr_matrix, data)

''
''
''
''
st.markdown("""
    <i style="font-size: 80%;">Created by Liam Fitzmaurice, Zhuonan Mai (Miranda), and Shance Zhao (Alex) at Massey University</i>
""", unsafe_allow_html=True)
