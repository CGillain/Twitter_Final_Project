from tkinter import *
import json
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import PercentFormatter
import yfinance as yf



def app():

    global root
    root= Tk()
    root.geometry("1400x800")
    root.title("Twitter Sentiment Analysis")
    root.configure(bg= 'white')

    window =Frame(root)
    #window.title("Zencheck Sentiment Analyzer")
    window.pack()


    #Options
    options= [
        "Airbus",
        "Boeing",
        "Coca-Cola",
        "EasyJet",
        "McDonalds",
        "Nike"]

    #updates text
    def option_handle():
        selected = var.get()
        if selected == "Airbus":
            return "Data_Airbus.txt"
        elif selected == "Boeing":
            return "Data_Airbus.txt"
        elif selected == "Coca-Cola":
            return "Data_CocaCola.txt"
        elif selected == "EasyJet":
            return "Data_EasyJet.txt"
        elif selected == "McDonalds":
            return "Data_Mcdonalds.txt"
        elif selected == "Nike":
            return "Data_Nike.txt"

    #updates stock
    def stock():
        selected = var.get()
        if selected == "Airbus":
            return "AIR.PA"
        elif selected == "Boeing":
            return "BA"
        elif selected == "Coca-Cola":
            return "KO"
        elif selected == "EasyJet":
            return "EZJ.L"
        elif selected == "McDonalds":
            return "MCD"
        elif selected == "Nike":
            return "NKE"


    def plot_graphs():


        # Initialize empty list to store tweets
        company_data = []

        # Open connection to file
        with open(option_handle(), "r") as tweets_file:
            # Read in tweets and store in list
            for line in tweets_file:
                tweet = json.loads(line)
                company_data.append(tweet)

        df_company_data = pd.DataFrame(company_data,
                                  columns=['created_at', 'lang', 'text', 'favorite_count', 'retweet_count',
                                           'reply_count'])
        # print(df_netflix)

        # Clean - wanted to keep the emojis
        # remove special characters and convert to lowercase
        df_company_data['text'] = df_company_data['text'].apply(lambda x: re.sub('[-!@#$:).;,?&]', '', x.lower()))
        df_company_data['text'] = df_company_data['text'].apply(lambda x: re.sub('  ', ' ', x))
        text = ' '.join(txt for txt in df_company_data.text)

        company_list = df_company_data['text'].tolist()

        # Sentiment analysis
        analyser = SentimentIntensityAnalyzer()
        company_data_sentiment = []
        for tweet in company_list:
            score = analyser.polarity_scores(tweet)
            tweeted = dict({"text": tweet})
            score.update(tweeted)
            company_data_sentiment.append(score)

        # Data Frame
        df_company_sentiment = pd.DataFrame(company_data_sentiment)
        df_company_sentiment = pd.concat(
            [df_company_sentiment,
             df_company_data["created_at"],
             df_company_data["retweet_count"],
             df_company_data["reply_count"],
             df_company_data["favorite_count"]],
            axis=1)

        #  Company Name
        df_company_sentiment.insert(5, "Company", var.get(), True)

        # Removing Duplicates
        df_company_sentiment = df_company_sentiment.drop_duplicates(subset=['text'], keep='first')

        # CHangine date fromat
        df_company_sentiment['created_at'] = pd.to_datetime(df_company_sentiment['created_at'])
        df_company_sentiment['created_at'] = df_company_sentiment['created_at'].dt.strftime('%Y-%m-%d')


        # Getting most negative tweets
        aggr_index = df_company_sentiment
        aggr_index.set_index('created_at', inplace=True)
        neg_df = []
        for index, row in aggr_index.iterrows():
            if row['compound'] < -0.4:
                neg_df.append(row)

        neg_df = pd.DataFrame(neg_df)
        print(neg_df)
        print(len(neg_df))

        # Aggregate per day mean of compound
        date = aggr_index.index
        aggregate = df_company_sentiment.groupby(date).compound.mean()
        aggregate = pd.DataFrame(aggregate)


        # assembling overall sentiment column
        sentiment_analysis = df_company_sentiment

        # create one column for sentiment
        sentiment_analysis["sentiment"] = ""
        for index, row in sentiment_analysis.iterrows():
            if row['compound'] < 0:
                sentiment_analysis.loc[index, 'sentiment'] = "negative"
            elif row['compound'] > 0:
                sentiment_analysis.loc[index, 'sentiment'] = "positive"
            else:
                sentiment_analysis.loc[index, 'sentiment'] = "neutral"

        # Getting data dates for stock price
        start = aggregate.first_valid_index()
        end = aggregate.last_valid_index()

        # Plotting
        # Plot of overall sentiment for the entire week - positive, negative, neutrl
        figure1 = plt.Figure()
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, root)
        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        ax = sentiment_analysis['sentiment'].value_counts(normalize=True).plot(kind='bar',
                                                                                    figsize=(8, 4),
                                                                        title="Overall Week Sentiment " + var.get() + " from " + start + " to " + end, color='y', ax= ax1)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Frequency")
        ax.patch.set_facecolor('black')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.annotate('{:.0%}'.format(height), (x, y + height + 0.01), color='w')
        plt.gcf().autofmt_xdate()


        # Getting Stock prices
        df = yf.download(stock(), start=start, end=end)
        df = pd.DataFrame(df)

        # Mergin the data to include stock price by date
        df_merge = aggregate.join(df, how='outer')
        df_merge = df_merge[df_merge['compound'].notna()]

        # Getting correlation
        d = df_merge.index
        corr = df_merge['compound'].corr(df_merge['Adj Close'])


        ## Plotting compound sentiment by day over stock price
        # Specify facecolor when creating the figure
        fig, ax2 = plt.subplots(figsize=(8, 4), dpi=100)
        bar2 = FigureCanvasTkAgg(fig, root)
        bar2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        ax2.set_xlabel('''Date
        
        Correlation is  '''  + str(corr))
        ax2.set_ylabel('Percentage Sentiment', color='blue')
        ax2.bar(d, df_merge['compound'], color='blue')
        ax2.set_title(var.get() +' Compound sentiment per day VS. Stock price' + " from " + start + " to " + end)
        ax2.tick_params(axis='y', labelcolor='blue')

        ax2.patch.set_facecolor('black')
        ax2.yaxis.set_major_formatter(PercentFormatter(1))
        for p in ax2.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax2.annotate('{:.0%}'.format(height), (x, y + height + 0.001), color='w')


        ax3 = ax2.twinx()

        ax3.set_ylabel('Close Price', color='red')
        ax3.plot(d, df_merge['Adj Close'], color='red', label= var.get()+' Stock price')
        ax3.tick_params(axis='y', labelcolor='red')
        ax3.legend(loc= 'best')

        fig.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.subplots_adjust(bottom=0.25)


    # Getting excel file of negative tweets

    def export():
        # Initialize empty list to store tweets
        # Initialize empty list to store tweets
        company_data = []

        # Open connection to file
        with open(option_handle(), "r") as tweets_file:
            # Read in tweets and store in list
            for line in tweets_file:
                tweet = json.loads(line)
                company_data.append(tweet)

        df_company_data = pd.DataFrame(company_data,
                                       columns=['created_at', 'lang', 'text', 'favorite_count', 'retweet_count',
                                                'reply_count'])
        # print(df_netflix)

        # Clean - wanted to keep the emojis
        # remove special characters and convert to lowercase
        df_company_data['text'] = df_company_data['text'].apply(lambda x: re.sub('[-!@#$:).;,?&]', '', x.lower()))
        df_company_data['text'] = df_company_data['text'].apply(lambda x: re.sub('  ', ' ', x))
        text = ' '.join(txt for txt in df_company_data.text)

        company_list = df_company_data['text'].tolist()

        # Sentiment analysis
        analyser = SentimentIntensityAnalyzer()
        company_data_sentiment = []
        for tweet in company_list:
            score = analyser.polarity_scores(tweet)
            tweeted = dict({"text": tweet})
            score.update(tweeted)
            company_data_sentiment.append(score)

        # as data frame
        df_company_sentiment = pd.DataFrame(company_data_sentiment)
        df_company_sentiment = pd.concat(
            [df_company_sentiment, df_company_data["created_at"], df_company_data["retweet_count"],
             df_company_data["reply_count"],
             df_company_data["favorite_count"]], axis=1)
        df_company_sentiment.insert(5, "Company", var.get(), True)
        df_company_sentiment = df_company_sentiment.drop_duplicates(subset=['text'], keep='first')

        # change date fromat
        df_company_sentiment['created_at'] = pd.to_datetime(df_company_sentiment['created_at'])
        df_company_sentiment['created_at'] = df_company_sentiment['created_at'].dt.strftime('%Y-%m-%d')
        # print(netflix_sentiment['created_at'])

        # Most negative tweets
        aggr_index = df_company_sentiment
        aggr_index.set_index('created_at', inplace=True)
        neg_df = []
        for index, row in aggr_index.iterrows():
            if row['compound'] < -0.5:
                neg_df.append(row)

        neg_df = pd.DataFrame(neg_df)
        neg_df.to_excel("NegativeTweets_" + var.get()+".xlsx", index=False)
        display.config(text=var.get() + " Excel file will appear in your file directory")


    #Drop Down Menu

    # Titel
    Zencheck= Label(window, text= "Zencheck Twitter Sentiment Analysis")
    Zencheck.config(font=("Calibri", 28))
    Zencheck.config(width=40)
    Zencheck.pack(side= TOP)

    #DropDown Menu
    var = StringVar()
    var.set(options[0])
    drop = OptionMenu(window, var, *options)
    drop.config(width=30)
    drop.pack(side= LEFT)

    # Plotting data Button
    runbtn = Button(window, text = 'Plot Data', command = plot_graphs)
    runbtn.config(width=30)
    runbtn.pack(side= LEFT)

    # Refresh button
    refresh_button = Button(window, text="Clear Data", command=clear)
    refresh_button.config(width=30)
    refresh_button.pack(side=LEFT)

    # Excel table button
    xlsx_button= Button(window, text="Get Negative tweets" , command=export)
    xlsx_button.config(width=30)
    xlsx_button.pack(side= LEFT )

    # Saying when table is loaded
    display = Label(window)
    display.pack(side=BOTTOM)

    root.mainloop()


if __name__ == '__main__':
    def clear():
        root.destroy()
        app()
    app()