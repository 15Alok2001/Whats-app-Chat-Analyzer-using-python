from collections import Counter
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
import nltk

extractor = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))
    return num_messages, len(words), num_media_messages, len(links)


def most_active(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'}
    )
    return x, df


def create_wc(selected_user, df):
    f = open('hinglish_words.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['temp_message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=""))
    return df_wc


def most_common_words(selected_user, df):
    f = open('hinglish_words.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    common_df = pd.DataFrame(Counter(words).most_common(20))
    return common_df


'''def emoji_counter(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.is_emoji['en']])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emojis'''


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline


def sentiment_Analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    new_df = pd.DataFrame(df, columns=["Date", "Time", "contact", "Message"])
    df['date'] = pd.to_datetime(df['date'])
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    # def remove_stop_words(message):
    #     y = []
    #     for word in message.lower().split():
    #         if word not in stop_words:
    #             y.append(word)
    #     return " ".join(y)

    temp = df.dropna()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sentiments = SentimentIntensityAnalyzer()
    temp["positive"] = [sentiments.polarity_scores(i)["pos"] for i in temp["message"]]
    temp["negative"] = [sentiments.polarity_scores(i)["neg"] for i in temp["message"]]
    temp["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in temp["message"]]
    return temp
