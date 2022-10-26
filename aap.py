import streamlit as st
import preprocessor
import matplotlib.pylab as plt
from Support import fetch_stats, most_active, create_wc, most_common_words, monthly_timeline, daily_timeline, \
    sentiment_Analysis

st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    # fetch unique user
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show the Analysis", user_list)

    if st.sidebar.button("Start Analysis"):
        num_messages, words, num_media_messages, num_links = fetch_stats(selected_user, df)
        st.title("Show Analysis")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Total Media File")
            st.title(num_media_messages)
        with  col4:
            st.header("Total Links")
            st.title(num_links)
        # monthly timeline
        st.title("Monthly Timeline")
        timeline = monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        if selected_user == 'Overall':
            st.title("Most Busy User")
            x, new_df = most_active(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)
        st.title("Word Cloud")
        df_wc = create_wc(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        most_common_df = most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title("Most Common Words")
        st.pyplot(fig)

        # emoji_df = emoji_counter(selected_user,df)
    #  st.dataframe(emoji_df)
        st.title("Sentiment Analysis")
        ana_df = sentiment_Analysis(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(ana_df)
