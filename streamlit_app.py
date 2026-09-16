import streamlit as st
import pandas as pd

st.set_page_config(page_title="F1 Podium Predictor", layout="wide")

st.title("F1 Podium Prediction", anchor=False)
st.caption("This model predicts which 3 drivers will finish on the podium in a given race, using only information **before** the race starts!")
with st.expander("Read more!", type='compact'):
    st.markdown('''
        **Features used in the final model:**
        - grid position + grid penalty
        - qualifying position and gap to pole
        - championship standings entering the race for both the driver & the team
        - recent form - average finish pos over the last 3 & 5 races
        - sprint race results - if sprint weekend
        - track volatility - by looking at how often the top-3 grid starters have historically held their place


        **Model:** Random Forest Classifier, trained on 2018-2023 seasons, and evaluated on held-out 2024-2025 unseen test set.

        **Versions:** Built model iteratively, for more info, see the Model Performance section
    ''')


@st.cache_data
def load_predictions():
    return pd.read_csv("f1_v3_predictions.csv")

@st.cache_data
def load_results():
    v1 = pd.read_csv("v1_model_results.csv")
    v2 = pd.read_csv("v2_model_results.csv")
    v3 = pd.read_csv("v3_final_locked_results.csv")
    return v1, v2, v3

df = load_predictions()
v1_results, v2_results, v3_results = load_results()

# ---------- Race explorer ----------
with st.container(border=True):
    st.header("Race Explorer", icon=":material/trophy:" ,anchor=False)
    st.caption("Browse individual races from the 2024–2025 held-out test set.")

    test = df[df['Year'] >= 2024].copy()

    all_races = (
        df[df['Year'] >= 2024][['Year', 'RoundNumber', 'EventName']]
        .drop_duplicates()
        .sort_values(['Year', 'RoundNumber'])
        .reset_index(drop=True)
    )

    # session state init
    if 'race_idx' not in st.session_state:
        st.session_state.race_idx = 0

    st.session_state.race_idx = max(0, min(st.session_state.race_idx, len(all_races) - 1))

    current = all_races.iloc[st.session_state.race_idx]
    current_year = current['Year']
    current_round = current['RoundNumber']

    # drop down for selecting year
    years = sorted(all_races['Year'].unique())
    selected_year = st.selectbox("Year", years, index=years.index(current_year), width=100)

    if selected_year != current_year:
        first_match = all_races[all_races['Year'] == selected_year].index[0]
        st.session_state.race_idx = first_match
        st.rerun()

    # drop down for selecting round
    rounds_this_year = all_races[all_races['Year'] == selected_year][['RoundNumber', 'EventName']].reset_index(drop=True)
    round_labels = rounds_this_year.apply(lambda r: f"R{r['RoundNumber']} -- {r['EventName']}", axis=1).tolist()

    current_label_idx = int(rounds_this_year[rounds_this_year['RoundNumber'] == current_round].index[0])
    selected_label = st.selectbox("Round", round_labels, index=current_label_idx, width=300)

    selected_round = int(selected_label.split(' -- ')[0].replace('R', ''))
    if selected_round != current_round:
        match = all_races[(all_races['Year'] == selected_year) & (all_races['RoundNumber'] == selected_round)].index[0]
        st.session_state.race_idx = match
        st.rerun()


    # prev + next buttons
    col_prev, col_next = st.columns(2, gap=None)
    with col_prev:
        if st.button("Prev", icon=":material/arrow_back:", disabled=st.session_state.race_idx == 0):
            st.session_state.race_idx -= 1
            st.rerun()

    with col_next:
        if st.button("Next", icon=":material/arrow_forward:"):
            st.session_state.race_idx += 1
            st.rerun()


    # recompute selected year + round
    current = all_races.iloc[st.session_state.race_idx]
    selected_year = current['Year']
    selected_round = current['RoundNumber']
    race = test[(test['Year'] == selected_year) & (test['RoundNumber'] == selected_round)]


    # st.divider()
    event_name = race['EventName'].iloc[0]
    st.subheader(event_name, anchor=False)
    st.caption("The model predicts *which  3 drivers* will podium - not their finishing order")

    # get actual
    actual_podium = race[race['Podium'] == 1][['Abbreviation', 'ClassifiedPosition']].sort_values('ClassifiedPosition').reset_index(drop=True)
    actual_podium['ClassifiedPosition'] = actual_podium['ClassifiedPosition'].astype(int)
    actual_podium.index = actual_podium.index + 1
    actual_podium.columns = ['Driver', 'Finish']

    # get predicted
    predicted_podium = race.nlargest(3, 'PodiumProb')[['Abbreviation', 'PodiumProb']].reset_index(drop=True)
    predicted_podium['PodiumProb'] = (predicted_podium['PodiumProb'] * 100).round(1).astype(str) + '%'
    predicted_podium.index = predicted_podium.index + 1
    predicted_podium.columns = ['Driver', 'Model Confidence']

    predicted_ids = set(predicted_podium['Driver'])
    actual_ids = set(actual_podium['Driver'])

    # comparison display
    def highlight_in_actual(row):
        color = 'background-color: #dcb36160' if row['Driver'] in actual_ids else ''
        return [color] * len(row)

    def highlight_in_predicted(row):
        color = 'background-color: #dcb36160' if row['Driver'] in predicted_ids else ''
        return [color] * len(row)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model's predicted top 3**")
        st.table(predicted_podium.style.apply(highlight_in_actual, axis=1))
    with col2:
        st.write("**Actual podium**")
        st.table(actual_podium.style.apply(highlight_in_predicted, axis=1))


    correct_picks = len(predicted_ids & actual_ids)
    st.write(f"Model correctly identified **{correct_picks} of 3** podium finishers for this race.")

    # what happened?
    missed_picks = race[race['Abbreviation'].isin(predicted_ids - actual_ids)][['Abbreviation', 'ClassifiedPosition', 'Status']]

    if not missed_picks.empty:
        st.write("**What happened to the drivers that the model predicted would stand on the podium?**")
        missed_display = missed_picks.copy()
        missed_display['Result'] = missed_display.apply(
            lambda r: r['Status'] if r['Status'] != 'Finished' else f"Finished P{int(r['ClassifiedPosition'])}",
            axis = 1
        )
        st.table(missed_display[['Abbreviation', 'Result']].rename(columns={'Abbreviation' : 'Driver'}),
                        hide_index=True, width=400, border="horizontal")


# ---------- Models ----------
with st.container(border=True):
    st.header("Model Performance", anchor=False)

    v1_final = v1_results[v1_results['model'] == 'best_rf_FINAL_TEST'].iloc[0]
    v2_final = v2_results[v2_results['model'] == 'best_rf_v2_FINAL_TEST'].iloc[0]
    v3_final = v3_results[v3_results['model'] == 'v3_track_LOCKED'].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("V1", anchor=False)
        st.caption("grid position + grid penalty")
        st.metric("F1 Score", f"{v1_final['f1']:.3f}")
        st.metric("ROC-AUC", f"{v1_final['roc_auc']:.3f}")
        st.metric("Top-3 Race Accuracy", f"{v1_final['top3_per_race_acc']:.3f}")
        

    with col2:
        st.subheader("V2", anchor=False)
        st.caption("\\+ standings, form, sprint data")
        st.metric("F1 Score", f"{v2_final['f1']:.3f}")
        st.metric("ROC-AUC", f"{v2_final['roc_auc']:.3f}")
        st.metric("Top-3 Race Accuracy", f"{v2_final['top3_per_race_acc']:.3f}")

    with col3:
        st.subheader("V3", anchor=False)
        st.caption("\\+ track volatility")
        st.metric("F1 Score", f"{v3_final['f1']:.3f}")
        st.metric("ROC-AUC", f"{v3_final['roc_auc']:.3f}")
        st.metric("Top-3 Race Accuracy", f"{v3_final['top3_per_race_acc']:.3f}")

    st.write("")
    with st.expander("About each model"):
            st.markdown('''
            - **V1:** The simplest possible baseline - only features are grid position + grid penalty. Used to establish
            whether richer features would actually help + reference point for later versions.
            - **V2:** Added championship standings, recent form, and sprint race data. These temporal features improved the model's overall ranking ability, though did not help
            with nailing the top-3.
            - **V3:** Added track volatility measuring how often top-3 grid starters have historically held their position at each circuit. This was a change that meaningfully improved top-3 
            race accuracy specifically, rather than just overall ranking ability. A second idea - adding within the race season was tested and found to hurt performance, so was dropped.
            ''')
    with st.expander("About metrics"):
        st.markdown('''
            - **F1 Score:** balances precision (when the model says "podium", how often is it correct) and recall ("how many actual podium finishers did the model catch")
            - **ROC-AUC:** how well does the model rank drivers by podium likelihood
            - **Top-3 Race Accuracy:** of the model's top 3 picks, how many actually stood on the podium
        ''')